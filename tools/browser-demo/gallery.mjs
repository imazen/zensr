// Enhance the retained, immutable gallery without rewriting its generated data.
const compare = (before, after, beforeLabel, afterLabel) =>
  "/?" +
  new URLSearchParams({
    reference: new URL(before, location.href).pathname,
    candidate: new URL(after, location.href).pathname,
    beforeLabel,
    afterLabel,
  });
const note = document.createElement("p");
note.textContent =
  "Click a candidate to inspect it against the HR reference with zoom, pan and flicker. The links below each row compare direct SR with repair-first SR.";
document.querySelector("h1").after(note);
for (const section of document.querySelectorAll("section")) {
  const anchors = [...section.querySelectorAll("figure a")],
    paths = anchors.map((a) => a.getAttribute("href"));
  for (let i = 1; i < anchors.length; i++) {
    const label = anchors[i].closest("figure").querySelector("figcaption")
      .firstChild.textContent;
    anchors[i].href = compare(paths[0], paths[i], "HR reference", label);
    anchors[i].title = "Open zoom/pan/flicker comparison";
  }
  const links = document.createElement("p");
  for (const [a, b, label] of [
    [2, 3, "SPANF"],
    [4, 5, "DSCF"],
  ]) {
    const link = document.createElement("a");
    link.href = compare(
      paths[a],
      paths[b],
      label + " direct",
      label + " after repair",
    );
    link.textContent = `Compare ${label}: direct ↔ repair-first`;
    link.style.cssText =
      "display:inline-block;margin:0 24px 8px 0;text-decoration:underline";
    links.append(link);
  }
  section.append(links);
}
