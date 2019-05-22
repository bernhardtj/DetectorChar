# DetectorChar
Detector Characterization tools.

## Noise Clustering

### [`travis`](https://gist.github.com/anchal-physics/c219a617293e9098b726bcb33692825f)`/ LaTeX` compilation [![Build Status](https://travis-ci.com/CaltechExperimentalGravity/DetectorChar.svg?token=uERMqbPZoxpPqfDGvg9v&branch=preparatory-work)](https://travis-ci.com/CaltechExperimentalGravity/DetectorChar)

- [Proposal](https://github.com/CaltechExperimentalGravity/DetectorChar/blob/gh-pages/reports/SURF2019/proposal/Proposal.pdf) 

## Summary pages

### Main scripts & condor

This repo includes the scripts behind the [40m summary pages](https://nodus.ligo.caltech.edu:30889/detcharsummary/). The main executable is `bin/gw_daily_summary_40m` which sets up and runs the jobs to produce HTML 
and plots for one day of data. This is the script that is run every 30 minutes 
by a cron-like Condor job. Such job is submited using the 
`condor/gw_daily_summary.sub` submit file. That is, you can "turn on" the pages 
by doing:

```
condor_submit DetectorChar/condor/gw_daily_summary.sub
```

Avoid doing this if job had already been submitted. You can check if this the 
case by looking at the Condor *queue*:

```
[40m@ldas-pcdev1 ~]$ condor_q | grep 40m
70387847.0   40m             7/22 15:49   0+00:18:23 I  0   219.7 gw_daily_summary_4
```

You can "turn off" the summary pages by *removing* the job (`condor_rm`) or by 
putting it on *hold* (`condor_hold`). In that case, you can then remove the hold by
*releasing* it (`condor_release`).

Note there are counterparts to `bin/gw_daily_summary_40m` and `condor/gw_daily_summary.sub` 
that do basically the same thing but for "yesterday's" data, namely `bin/gw_daily_summary_40m_rerun` 
and `condor/gw_daily_summary_rerun.sub`.

### Auxiliary scripts

The `bin/pushnodus` script syncs the output of the code back to nodus, while 
`bin/checkstatus` checks the health of the code. Both these scripts are run from 
crontabs installed in the 40m accout in ldas-pcdev1 at CIT.

### More info

- [Summary pages](https://wiki-40m.ligo.caltech.edu/DailySummaryHelp)
- [Condor](http://research.cs.wisc.edu/htcondor/manual/)
