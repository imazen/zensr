"""Exact paired scalar summaries, with explicit empty restored subsets."""
from decimal import Decimal as D
import statistics
from summarize import summary


def paired(cells, arm, baseline):
    if not cells:
        return dict(n=0, mean_gain=None, median_gain=None, harm=0, harm_percent=None,
            wins=0, ties=0, worse=0, worst_gain=None, worst_id=None,
            mean_butter_pnorm3_delta=None, median_butter_pnorm3_delta=None,
            butter_wins=0, butter_ties=0, butter_worse=0, worst_butter_delta=None)
    rows=[dict(id=c['row']['id'],score=c[arm]['ssim2'],baseline=c[baseline]['ssim2'],
        butter=c[arm]['butter_pnorm3'],baseline_butter=c[baseline]['butter_pnorm3']) for c in cells]
    result=summary(rows)
    gain=[D(r['score'])-D(r['baseline']) for r in rows]
    butter=[D(r['butter'])-D(r['baseline_butter']) for r in rows]
    result.update(median_gain=str(statistics.median(gain)),wins=sum(x>0 for x in gain),ties=sum(x==0 for x in gain),
        median_butter_pnorm3_delta=str(statistics.median(butter)),butter_wins=sum(x<0 for x in butter),
        butter_ties=sum(x==0 for x in butter),butter_worse=sum(x>0 for x in butter),worst_butter_delta=str(max(butter)))
    return result
