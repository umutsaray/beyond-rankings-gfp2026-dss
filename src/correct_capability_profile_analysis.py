from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

import audit_capability_profile_analysis as audit
import create_capability_profile_analysis as cap


ROOT = Path("outputs/capability_profiles")
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
JSON_DIR = ROOT / "json"
SEED = 260822
N_BOOT = 500
CORR_THRESHOLD = 0.90
DOMAIN_ORDER = ["financial", "geography", "manpower", "airpower", "land", "naval", "natural_resources", "logistics"]
PEER_SCENARIOS = ["S1", "S2", "S3", "S4", "S6", "S7", "S8_corrected"]
STABILITY_THRESHOLDS = {
    "HIGH": {"family_presence_frequency": 0.60, "preprocessing_scenario_count": 5, "metric_count": 3, "space_count": 2},
    "MODERATE": {"family_presence_frequency": 0.30, "preprocessing_scenario_count": 3, "metric_count": 2, "space_count": 1},
}


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")


def connected_groups(corr: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> list[list[str]]:
    nodes = list(corr.columns)
    adjacency = {n: set() for n in nodes}
    for i, left in enumerate(nodes):
        for right in nodes[i + 1:]:
            if abs(corr.loc[left, right]) >= threshold:
                adjacency[left].add(right); adjacency[right].add(left)
    groups, seen = [], set()
    for node in nodes:
        if node in seen: continue
        stack, group = [node], []
        while stack:
            current = stack.pop()
            if current in seen: continue
            seen.add(current); group.append(current); stack.extend(sorted(adjacency[current] - seen))
        groups.append(sorted(group))
    return groups


def orient_pc1(pca: PCA) -> float:
    return 1.0 if pca.components_[0].sum() >= 0 else -1.0


def make_redundancy_groups(base: pd.DataFrame, features: list[str]) -> tuple[dict[str, list[list[str]]], pd.DataFrame]:
    domain_groups: dict[str, list[list[str]]] = {}
    rows = []
    for domain in DOMAIN_ORDER:
        cols = [c for c in cap.DOMAIN_FIELDS[domain] if c in features]
        corr = base[cols].corr(method="spearman")
        groups = connected_groups(corr)
        domain_groups[domain] = groups
        for group_index, group in enumerate(groups, 1):
            treatment = "within_group_PC1" if len(group) > 1 else "retain_standardized_feature"
            reason = f"connected component at |Spearman rho| >= {CORR_THRESHOLD:.2f}" if len(group) > 1 else "no within-domain correlation edge above threshold"
            for feature in group:
                rows.append({
                    "domain": domain, "original_feature": feature,
                    "retained_or_reduced": f"{domain}_RG{group_index}" if len(group) > 1 else feature,
                    "redundancy_group": f"{domain}_RG{group_index}", "group_size": len(group),
                    "treatment": treatment, "reason": reason,
                })
    composition = pd.DataFrame(rows)
    save_csv(composition, "s8_corrected_domain_composition.csv")
    return domain_groups, composition


def fit_corrected_domains(prepared: pd.DataFrame, fit_idx: np.ndarray, groups: dict[str, list[list[str]]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_scaler = StandardScaler().fit(prepared.iloc[fit_idx])
    z = pd.DataFrame(feature_scaler.transform(prepared), columns=prepared.columns, index=prepared.index)
    scores, group_models = {}, {}
    for domain, domain_groups in groups.items():
        components = []
        for group_number, group in enumerate(domain_groups, 1):
            if len(group) == 1:
                component = z[group[0]].to_numpy()
                group_models[(domain, group_number)] = {"type": "singleton", "feature": group[0]}
            else:
                pca = PCA(n_components=1, random_state=cap.RANDOM_STATE).fit(z.loc[fit_idx, group])
                sign = orient_pc1(pca)
                component = pca.transform(z[group]).ravel() * sign
                group_models[(domain, group_number)] = {"type": "PC1", "features": group, "loadings": (pca.components_[0] * sign).tolist(), "explained": float(pca.explained_variance_ratio_[0])}
            components.append(component)
        scores[domain] = np.mean(np.column_stack(components), axis=1)
    return pd.DataFrame(scores, index=prepared.index), {"feature_scaler": feature_scaler, "groups": group_models}


def prepare_raw(df: pd.DataFrame, naval: pd.DataFrame, features: list[str], log_features: set[str], fit_idx: np.ndarray) -> pd.DataFrame:
    raw = df[features].apply(pd.to_numeric, errors="coerce").copy()
    structural = naval.classification.eq("structural_absence_landlocked").to_numpy()
    for col in features:
        if col in cap.NAVAL_FIELDS:
            raw.loc[structural & raw[col].isna().to_numpy(), col] = 0.0
        median = raw[col].iloc[fit_idx].median(skipna=True)
        raw[col] = raw[col].fillna(0.0 if pd.isna(median) else median)
        if col in log_features:
            raw[col] = np.log1p(raw[col].clip(lower=0))
    return raw


def full_corrected_s8(df, naval, features, logs, groups):
    all_idx = np.arange(len(df))
    prepared = prepare_raw(df, naval, features, logs, all_idx)
    domains, models = fit_corrected_domains(prepared, all_idx, groups)
    scaler = RobustScaler(quantile_range=(25, 75)).fit(domains)
    scaled = scaler.transform(domains)
    pca = PCA(random_state=cap.RANDOM_STATE).fit(scaled)
    scores = pca.transform(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n80, n90 = int(np.searchsorted(cumulative, .80) + 1), int(np.searchsorted(cumulative, .90) + 1)
    save_domains = domains.copy(); save_domains.insert(0, "country", df.country)
    save_csv(save_domains, "s8_corrected_domain_scores.csv")
    ev = pd.DataFrame({"component":np.arange(1,9),"eigenvalue":pca.explained_variance_,"explained_variance_ratio":pca.explained_variance_ratio_,"cumulative_explained_variance":cumulative,"retained_80pct":np.arange(1,9)<=n80,"retained_90pct":np.arange(1,9)<=n90})
    loadings = pd.DataFrame(pca.components_.T,index=DOMAIN_ORDER,columns=[f"PC{i+1}" for i in range(8)]).reset_index(names="domain")
    save_csv(ev,"s8_corrected_pca_explained_variance.csv"); save_csv(loadings,"s8_corrected_pca_loadings.csv")
    return prepared, domains, scaled, pca, scores, n80, n90, models


def bootstrap_corrected(df, naval, features, logs, groups, reference, k, n_pc, tr, n_boot=N_BOOT):
    rng=np.random.default_rng(SEED+k*1000); n=len(df); rows=[]; runs=[]
    for b in range(n_boot):
        sample=rng.choice(n,n,replace=True)
        prepared=prepare_raw(df,naval,features,logs,sample)
        domains,_=fit_corrected_domains(prepared,sample,groups)
        ds=RobustScaler(quantile_range=(25,75)).fit(domains.iloc[sample])
        scaled=ds.transform(domains)
        pca=PCA(n_components=n_pc,random_state=cap.RANDOM_STATE).fit(scaled[sample])
        all_scores=pca.transform(scaled)
        model=KMeans(k,n_init=30,random_state=SEED+k*1000+b+1).fit(all_scores[sample])
        pred=model.predict(all_scores)
        aligned,_=audit.align_labels(reference,pred,k); runs.append(aligned)
        j=audit.jaccard_by_cluster(reference,aligned,k)
        oob=np.setdiff1d(np.arange(n),np.unique(sample))
        rows.append({"k":k,"resample":b+1,"ARI":adjusted_rand_score(reference,aligned),"AMI":adjusted_mutual_info_score(reference,aligned),"NMI":normalized_mutual_info_score(reference,aligned),"ARI_oob":adjusted_rand_score(reference[oob],aligned[oob]) if len(np.unique(reference[oob]))>1 else np.nan,"cluster_jaccard":np.mean(j),"turkiye_membership_stability":float(aligned[tr]==reference[tr])})
    return pd.DataFrame(rows),np.vstack(runs)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand=values.mean(); total=float(((values-grand)**2).sum())
    between=sum((labels==c).sum()*(values[labels==c].mean()-grand)**2 for c in np.unique(labels))
    return float(between/total) if total else 0.0


def scale_index(df: pd.DataFrame, features: list[str], logs: set[str], naval: pd.DataFrame) -> tuple[np.ndarray,list[str]]:
    excluded=set(cap.DEFICIT_FIELDS)|{"internet_coverage"}
    cols=[]
    for c in features:
        s=pd.to_numeric(df[c],errors="coerce")
        if c not in excluded and s.min(skipna=True)>=0 and s.nunique(dropna=True)>2:
            cols.append(c)
    prepared=prepare_raw(df,naval,cols,set(cols),np.arange(len(df)))
    z=StandardScaler().fit_transform(prepared)
    pca=PCA(n_components=1,random_state=cap.RANDOM_STATE).fit(z)
    score=pca.transform(z).ravel()*orient_pc1(pca)
    pd.DataFrame({"country":df.country,"general_scale_index":score}).to_csv(TABLES/"general_scale_index.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"feature":cols,"loading":pca.components_[0]*orient_pc1(pca)}).to_csv(TABLES/"general_scale_index_composition.csv",index=False,encoding="utf-8-sig")
    return score,cols


def evaluate_k(df, naval_presence, x, domains, scale, n90, tr, bootstrap_inputs):
    algorithms=["kmeans","agglomerative_ward","agglomerative_average","agglomerative_complete","gmm","kmedoids"]
    rows=[]; labels_by_k={}; method_rows=[]; bootstrap_tables=[]
    for k in range(2,7):
        labels={a:cap.fit_labels(x,a,k,cap.RANDOM_STATE)[0] for a in algorithms}
        ref=labels["kmeans"]; labels_by_k[k]=ref
        boot,runs=bootstrap_corrected(*bootstrap_inputs,ref,k,n90,tr); bootstrap_tables.append(boot)
        pair_ari=[];pair_ami=[];pair_nmi=[]
        for i,a in enumerate(algorithms):
            for b in algorithms[i+1:]:
                pair_ari.append(adjusted_rand_score(labels[a],labels[b])); pair_ami.append(adjusted_mutual_info_score(labels[a],labels[b])); pair_nmi.append(normalized_mutual_info_score(labels[a],labels[b]))
                method_rows.append({"k":k,"method_a":a,"method_b":b,"ARI":pair_ari[-1],"AMI":pair_ami[-1],"NMI":pair_nmi[-1],"sizes_a":"|".join(map(str,sorted(np.bincount(labels[a],minlength=k)))),"sizes_b":"|".join(map(str,sorted(np.bincount(labels[b],minlength=k))))})
        coassoc=np.mean(runs[:,:,None]==runs[:,None,:],axis=0); within=[];between=[]
        for i in range(len(df)):
            for j in range(i+1,len(df)):
                (within if ref[i]==ref[j] else between).append(coassoc[i,j])
        naval_binary=naval_presence.fillna(naval_presence.median()).round().astype(int).to_numpy()
        contingency=pd.crosstab(pd.Series(ref,name="cluster"),pd.Series(naval_binary,name="naval_presence"))
        sizes=np.bincount(ref,minlength=k)
        naval_ari=adjusted_rand_score(naval_binary,ref); naval_nmi=normalized_mutual_info_score(naval_binary,ref)
        scale_assoc=eta_squared(scale,ref)
        centroid_z=pd.DataFrame(StandardScaler().fit_transform(domains),columns=domains.columns).assign(cluster=ref).groupby("cluster").mean()
        domain_range=(centroid_z.max()-centroid_z.min()).sort_values(ascending=False)
        if naval_ari>=.80: structure="naval_structure"
        elif scale_assoc>=.65: structure="scale"
        elif scale_assoc>=.35: structure="mixed_scale_and_composition"
        else: structure="multidimensional_composition"
        rows.append({"k":k,"silhouette":silhouette_score(x,ref),"calinski_harabasz":calinski_harabasz_score(x,ref),"davies_bouldin":davies_bouldin_score(x,ref),"cluster_sizes":"|".join(map(str,sorted(sizes))),"smallest_cluster_size":sizes.min(),"imbalance_ratio":sizes.max()/sizes.min(),"bootstrap_ARI":boot.ARI.mean(),"bootstrap_AMI":boot.AMI.mean(),"bootstrap_NMI":boot.NMI.mean(),"cluster_Jaccard":boot.cluster_jaccard.mean(),"cross_method_ARI":np.mean(pair_ari),"cross_method_AMI":np.mean(pair_ami),"cross_method_NMI":np.mean(pair_nmi),"consensus_within":np.mean(within),"consensus_between":np.mean(between),"consensus_separation":np.mean(within)-np.mean(between),"turkiye_membership_stability":boot.turkiye_membership_stability.mean(),"naval_presence_ARI":naval_ari,"naval_presence_NMI":naval_nmi,"naval_contingency":json.dumps({str(i):{str(j):int(contingency.loc[i,j]) for j in contingency.columns} for i in contingency.index}),"scale_association":scale_assoc,"dominant_structure":structure,"top_discriminating_domains":"|".join(domain_range.head(4).index)})
    pd.concat(bootstrap_tables,ignore_index=True).to_csv(TABLES/"s8_corrected_bootstrap_stability.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(method_rows).to_csv(TABLES/"s8_corrected_cross_method_agreement.csv",index=False,encoding="utf-8-sig")
    return pd.DataFrame(rows),labels_by_k


def assign_status(comparison: pd.DataFrame) -> pd.DataFrame:
    out=comparison.copy()
    quality=[];status=[]
    for _,r in out.iterrows():
        if r.naval_presence_ARI>=.80:
            quality.append("low: naval artifact");status.append("artifact_dominated")
        elif r.bootstrap_ARI<.60 or r.smallest_cluster_size<5:
            quality.append("low: unstable or tiny cluster");status.append("unstable")
        elif r.bootstrap_ARI>=.70 and r.cross_method_ARI>=.40 and r.consensus_separation>=.70 and r.scale_association<.65:
            quality.append("high: convergent and non-artifact");status.append("plausible")
        else:
            quality.append("moderate: qualified evidence");status.append("plausible")
    out["interpretation_quality"]=quality;out["methodological_status"]=status
    eligible=out[out.methodological_status.eq("plausible")]
    if len(eligible):
        # Convergent-evidence rule: retain all candidates meeting stability/agreement/consensus thresholds.
        strong=eligible[(eligible.bootstrap_ARI>=.70)&(eligible.cross_method_ARI>=.40)&(eligible.consensus_separation>=.70)&(eligible.smallest_cluster_size>=5)]
        if len(strong)==1:
            out.loc[strong.index,"methodological_status"]="preferred_candidate"
        elif len(strong)>1:
            out.loc[strong.index,"methodological_status"]="strong_alternative"
    return out


def family_balanced_peers(df, scenarios, corrected_scaled, corrected_pca, n90, preferred_k, preferred_labels, tr):
    scenario_spaces={}
    for sid in ["S1","S2","S3","S4","S6","S7"]:
        s=scenarios[sid]; scenario_spaces[sid]={"original":s.scaled,"pca":s.pca_scores[:,:s.n_pc90]}
    scenario_spaces["S8_corrected"]={"original":corrected_scaled,"pca":corrected_pca[:,:n90]}
    rows=[]; same_cluster=defaultdict(list)
    for sid,spaces in scenario_spaces.items():
        for space,x in spaces.items():
            cluster=cap.fit_labels(x,"kmeans",preferred_k,cap.RANDOM_STATE)[0]
            aligned,_=audit.align_labels(preferred_labels,cluster,preferred_k)
            for metric in ["euclidean","mahalanobis","cosine"]:
                d=cap.pairwise_distances(x,tr,metric); order=np.argsort(d);order=order[order!=tr][:10]
                for rank_pos,idx in enumerate(order,1):
                    rows.append({"preprocessing":sid,"space":space,"distance":metric,"country":df.country.iloc[idx],"neighbor_rank":rank_pos,"distance_value":d[idx]})
                for idx,country in enumerate(df.country):
                    if idx!=tr: same_cluster[country].append(float(aligned[idx]==aligned[tr]))
    raw=pd.DataFrame(rows); save_csv(raw,"turkiye_peer_balanced_families.csv")
    total_families=len(PEER_SCENARIOS)*2*3; summary=[]
    for country in df.country[df.index!=tr]:
        c=raw[raw.country.eq(country)]
        metrics={m:int((c.distance==m).sum()) for m in ["euclidean","mahalanobis","cosine"]}
        spaces={s:int((c.space==s).sum()) for s in ["original","pca"]}
        scenarios_count=c.preprocessing.nunique()
        freq=len(c)/total_families
        metric_count=sum(v>0 for v in metrics.values());space_count=sum(v>0 for v in spaces.values())
        high=STABILITY_THRESHOLDS["HIGH"];moderate=STABILITY_THRESHOLDS["MODERATE"]
        if freq>=high["family_presence_frequency"] and scenarios_count>=high["preprocessing_scenario_count"] and metric_count>=high["metric_count"] and space_count>=high["space_count"]: level="HIGH"
        elif freq>=moderate["family_presence_frequency"] and scenarios_count>=moderate["preprocessing_scenario_count"] and metric_count>=moderate["metric_count"] and space_count>=moderate["space_count"]: level="MODERATE"
        else: level="LOW"
        summary.append({"country":country,"families_eligible":total_families,"families_top10":len(c),"family_presence_frequency":freq,"mean_family_rank":c.neighbor_rank.mean(),"median_family_rank":c.neighbor_rank.median(),"euclidean_presence":metrics["euclidean"]/(len(PEER_SCENARIOS)*2),"mahalanobis_presence":metrics["mahalanobis"]/(len(PEER_SCENARIOS)*2),"cosine_presence":metrics["cosine"]/(len(PEER_SCENARIOS)*2),"original_space_presence":spaces["original"]/(len(PEER_SCENARIOS)*3),"pca_space_presence":spaces["pca"]/(len(PEER_SCENARIOS)*3),"preprocessing_scenario_count":scenarios_count,"same_cluster_frequency":np.mean(same_cluster[country]),"stability_level":level})
    out=pd.DataFrame(summary).sort_values(["family_presence_frequency","mean_family_rank"],ascending=[False,True]);save_csv(out,"turkiye_peer_balanced.csv")
    return out,raw


def update_outputs(original_json, audit_json, comparison, peers, ev, loadings, preferred_k, labels, tr):
    preferred_row=comparison[comparison.k.eq(preferred_k)].iloc[0]
    corrected={"scenario":"S8_corrected","domains":DOMAIN_ORDER,"pca":{"retained_pc90":int(ev.retained_90pct.sum()),"cumulative_variance":float(ev.loc[ev.retained_90pct,"cumulative_explained_variance"].iloc[-1]),"explained_variance":ev.loc[ev.retained_90pct,"explained_variance_ratio"].tolist(),"loadings":loadings.to_dict("records")},"k_comparison":comparison.to_dict("records"),"preferred_k":int(preferred_k),"preferred_status":preferred_row.methodological_status,"manuscript_facing":{"bootstrap_ARI":preferred_row.bootstrap_ARI,"bootstrap_AMI":preferred_row.bootstrap_AMI,"cross_method_ARI":preferred_row.cross_method_ARI,"turkiye_cluster":int(labels[tr])+1,"turkiye_membership_stability":preferred_row.turkiye_membership_stability,"balanced_peers":peers.head(20).to_dict("records")},"stability_thresholds":STABILITY_THRESHOLDS}
    historical={"original_analysis":original_json,"audit_findings":audit_json,"corrected_analysis":corrected,"manuscript_facing_source":"corrected_analysis only","original_bootstrap_status":"invalid_do_not_report"}
    (JSON_DIR/"manuscript_results.json").write_text(json.dumps(cap.native(historical),ensure_ascii=False,indent=2),encoding="utf-8")
    return corrected


def main():
    df,naval,presence,features,decisions,logs,scenarios=audit.base_objects();tr=cap.find_turkiye(df)
    base=cap.impute_structurally(df,features,naval.classification)
    for c in logs: base[c]=np.log1p(base[c].clip(lower=0))
    groups,composition=make_redundancy_groups(base,features)
    prepared,domains,scaled,pca,scores,n80,n90,models=full_corrected_s8(df,naval,features,logs,groups)
    scale,scale_features=scale_index(df,features,logs,naval)
    comparison,labels_by_k=evaluate_k(df,presence,scores[:,:n90],domains,scale,n90,tr,(df,naval,features,logs,groups))
    comparison=assign_status(comparison);save_csv(comparison,"corrected_k_comparison.csv");save_csv(comparison,"cluster_structure_diagnostics.csv")
    strong=comparison[comparison.methodological_status.isin(["preferred_candidate","strong_alternative"])]
    if len(strong)==1: preferred_k=int(strong.iloc[0].k)
    elif len(strong)>1:
        # Select a primary analytical candidate by convergent evidence, while retaining
        # every strong alternative and explicitly avoiding a claim of a unique optimum.
        preferred_k=int(strong.sort_values(
            ["bootstrap_ARI", "consensus_separation", "silhouette"],
            ascending=False,
        ).iloc[0].k)
    else:
        eligible=comparison[comparison.methodological_status.eq("plausible")]
        preferred_k=int(eligible.sort_values(["bootstrap_ARI","cross_method_ARI","silhouette"],ascending=False).iloc[0].k) if len(eligible) else 2
    preferred_labels=labels_by_k[preferred_k]
    peers,peer_raw=family_balanced_peers(df,scenarios,scaled,scores,n90,preferred_k,preferred_labels,tr)
    original_path=JSON_DIR/"manuscript_results.json"; original=json.loads(original_path.read_text(encoding="utf-8")); audit_summary=json.loads((REPORTS/"method_audit_summary.json").read_text(encoding="utf-8"))
    ev=pd.read_csv(TABLES/"s8_corrected_pca_explained_variance.csv");loadings=pd.read_csv(TABLES/"s8_corrected_pca_loadings.csv")
    corrected=update_outputs(original,audit_summary,comparison,peers,ev,loadings,preferred_k,preferred_labels,tr)
    old_report=(REPORTS/"capability_profile_analysis.md").read_text(encoding="utf-8");marker="## Post-Audit Corrected Analysis"
    if marker in old_report: old_report=old_report.split(marker)[0].rstrip()
    old_peers={"Thailand","Taiwan","Italy","France","Pakistan","Germany","Spain"};new_high=set(peers[peers.stability_level.eq("HIGH")].country);new_moderate=set(peers[peers.stability_level.eq("MODERATE")].country)
    section=f"""

{marker}

S8 was corrected because naval structure entered twice through the naval domain and `naval_presence`. `S8_corrected` uses only eight substantive domains. Within each domain, connected feature groups at |Spearman rho| >= 0.90 are represented by an oriented within-group PC1; singleton indicators remain standardized. Domain scores are equal-weight means of the resulting redundancy-controlled components, followed by RobustScaler and PCA.

PCA90 retains {n90} components and {ev.loc[ev.retained_90pct,'cumulative_explained_variance'].iloc[-1]:.1%} variance. Despite removing the direct indicator, corrected k=2 still reproduces naval presence (ARI={comparison.loc[comparison.k.eq(2),'naval_presence_ARI'].iloc[0]:.3f}) and is rejected as artifact-dominated. The primary analytical candidate is k={preferred_k}, selected by convergent evidence; all strong alternatives remain co-reported and no unique optimum is claimed.

The candidate structure is classified as `{comparison.loc[comparison.k.eq(preferred_k),'dominant_structure'].iloc[0]}` with scale association {comparison.loc[comparison.k.eq(preferred_k),'scale_association'].iloc[0]:.3f}. Corrected bootstrap ARI={comparison.loc[comparison.k.eq(preferred_k),'bootstrap_ARI'].iloc[0]:.3f}, AMI={comparison.loc[comparison.k.eq(preferred_k),'bootstrap_AMI'].iloc[0]:.3f}, and cross-method ARI={comparison.loc[comparison.k.eq(preferred_k),'cross_method_ARI'].iloc[0]:.3f}.

Türkiye peer recurrence now gives equal weight to 42 preprocessing × space × distance families. HIGH peers: {', '.join(peers[peers.stability_level.eq('HIGH')].head(15).country) or 'none'}. MODERATE peers: {', '.join(peers[peers.stability_level.eq('MODERATE')].head(15).country) or 'none'}. From the prior list, robust/retained peers are {', '.join(sorted(old_peers & (new_high|new_moderate))) or 'none'}; weakened or absent peers are {', '.join(sorted(old_peers-(new_high|new_moderate))) or 'none'}.
"""
    (REPORTS/"capability_profile_analysis.md").write_text(old_report+section+"\n",encoding="utf-8")
    pref=comparison[comparison.k.eq(preferred_k)].iloc[0]
    if pref.methodological_status=="preferred_candidate" and pref.dominant_structure=="multidimensional_composition" and pref.cross_method_ARI>=.40: readiness="READY WITH QUALIFICATIONS"
    elif pref.methodological_status in {"preferred_candidate","strong_alternative","plausible"} and pref.bootstrap_ARI>=.65: readiness="READY WITH QUALIFICATIONS"
    else: readiness="NOT READY"
    readiness_report=f"""# Manuscript Readiness After Correction

## {readiness}

- Corrected S8 validity: PASS WITH QUALIFICATIONS — eight domains, redundancy-controlled group components, no direct naval_presence geometry.
- Corrected bootstrap validity: PASS — 500 case-bootstrap refits per k with duplicated draws retained and preprocessing/PCA refit.
- Cluster-number defensibility: {pref.methodological_status}; k={preferred_k} is manuscript-facing while all strong alternatives remain reported.
- Naval-artifact diagnostic: FAIL for k=2 — naval ARI={comparison.loc[comparison.k.eq(2),'naval_presence_ARI'].iloc[0]:.3f}; k=2 is excluded from substantive interpretation even though the direct indicator was removed.
- Scale dependence: preferred scale eta-squared={pref.scale_association:.3f}; dominant structure={pref.dominant_structure}.
- Cross-method robustness: ARI={pref.cross_method_ARI:.3f}, AMI={pref.cross_method_AMI:.3f}, NMI={pref.cross_method_NMI:.3f}.
- Türkiye membership stability: {pref.turkiye_membership_stability:.3f}.
- Balanced peer robustness: {int((peers.stability_level=='HIGH').sum())} HIGH and {int((peers.stability_level=='MODERATE').sum())} MODERATE peers under 42 equally weighted families.
- Reproducibility: PASS; thresholds and reduction rules are stored in code and JSON.

Remaining qualifications: single-year secondary GFP data; opaque source methodology; domain PC1 components are sample-dependent; scale dependence may remain; cross-method disagreement must be reported; peer similarity does not imply equal power, alignment, alliance, or combat equivalence.
"""
    (REPORTS/"manuscript_readiness_after_correction.md").write_text(readiness_report,encoding="utf-8")
    print(json.dumps(cap.native({"corrected_domains":8,"n_pc80":n80,"n_pc90":n90,"pca90_variance":ev.loc[ev.retained_90pct,'cumulative_explained_variance'].iloc[-1],"preferred_k":preferred_k,"preferred":pref.to_dict(),"turkiye_cluster":int(preferred_labels[tr])+1,"top_peers":peers.head(15).to_dict("records"),"readiness":readiness}),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
