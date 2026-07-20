
def write_report(result,science,membership,out,posterior_summary=None):
    pb,pr,pbg=membership[:,0],membership[:,1],membership[:,2]
    lines=["# NGC 5128 three-component hierarchical fit v5","",
      f"- Fitted objects: **{len(science)}**",
      f"- Membership-weighted blue GCs: **{pb.sum():.1f}**",
      f"- Membership-weighted red GCs: **{pr.sum():.1f}**",
      f"- Membership-weighted contaminants: **{pbg.sum():.1f}**","",
      "## MAP parameters"]
    for k,v in result["parameters"].items():
        lines.append(f"- {k}: **{v:.5f}**")
    if posterior_summary:
        lines += ["","## MCMC intervals"]
        for k,v in posterior_summary.items():
            lines.append(f"- {k}: **{v['median']:.5f}** "
                         f"(-{v['median']-v['p16']:.5f}, +{v['p84']-v['median']:.5f})")
    lines += ["","## Notes",
      "- Blue GCs, red GCs, and contaminants are explicit competing components.",
      "- Blue and red GCs may have different mass functions and radial scales.",
      "- The contaminant density uses a Gaussian-mixture model.",
      "- NumPy arrays are passed consistently to scikit-learn, eliminating the feature-name warning.",
      "- Final scientific use still requires systematic runs."]
    (out/"RESULTS_REPORT_V4.md").write_text("\n".join(lines))
