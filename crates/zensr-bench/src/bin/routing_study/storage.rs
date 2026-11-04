use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

pub(super) fn publish(path: &Path, bytes: &[u8]) {
    publish_after_create(path, bytes, || {});
}

// Publish only fully written bytes. A hard link creates the canonical name
// atomically without overwriting an existing artifact from another worker.
fn publish_after_create(path: &Path, bytes: &[u8], after_create: impl FnOnce()) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let (staging, mut file) = loop {
        let staging = path.with_extension(format!(
            "partial-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
        {
            Ok(file) => break (staging, file),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("artifact staging {}: {e}", staging.display()),
        }
    };
    after_create();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
    drop(file);
    match fs::hard_link(&staging, path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            assert!(
                fs::read(path).unwrap() == bytes,
                "existing artifact changed"
            );
        }
        Err(e) => panic!("artifact {}: {e}", path.display()),
    }
    // The complete bytes are now retained at the canonical path. Failed writes
    // or conflicts leave their staging bytes intact for diagnosis.
    fs::remove_file(staging).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};

    fn destination() -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::var_os("TMPDIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join("tmp")
            });
        fs::create_dir_all(&root).unwrap();
        loop {
            let path = root.join(format!(
                "zensr-storage-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return path.join("artifact.bin"),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("test directory: {e}"),
            }
        }
    }

    #[test]
    fn concurrent_identical_publishers_all_observe_complete_bytes() {
        let path = Arc::new(destination());
        let bytes = Arc::new(vec![0xa7; 4 * 1024 * 1024]);
        let barrier = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let path = Arc::clone(&path);
                let bytes = Arc::clone(&bytes);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    publish(&path, &bytes);
                })
            })
            .collect();
        let passed: Vec<_> = threads.into_iter().map(|t| t.join().is_ok()).collect();
        assert!(
            passed.iter().all(|&ok| ok),
            "a concurrent publisher observed an incomplete artifact"
        );
        assert!(fs::read(&*path).unwrap() == *bytes);
    }

    #[test]
    fn publisher_paused_before_write_does_not_expose_partial_artifact() {
        let path = destination();
        let worker_path = path.clone();
        let (created_tx, created_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            publish_after_create(&worker_path, b"complete", || {
                created_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
            });
        });
        created_rx.recv().unwrap();
        let other = std::panic::catch_unwind(|| publish(&path, b"complete"));
        resume_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(other.is_ok(), "a publisher observed a partial artifact");
        assert_eq!(fs::read(path).unwrap(), b"complete");
    }

    #[test]
    fn conflicting_existing_artifact_is_not_overwritten() {
        let path = destination();
        fs::write(&path, b"existing").unwrap();
        assert!(std::panic::catch_unwind(|| publish(&path, b"different")).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"existing");
    }
}
