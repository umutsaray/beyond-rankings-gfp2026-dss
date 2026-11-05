from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler

import audit_capability_profile_analysis as audit
import correct_capability_profile_analysis as corr
import create_capability_profile_analysis as cap


def main() -> None:
    df, naval, _presence, features, _decisions, logs, _scenarios = audit.base_objects()
    base = cap.impute_structurally(df, features, naval.classification)
    for column in logs:
        base[column] = np.log1p(base[column].clip(lower=0))

    rows = []
    labels_by_threshold = {}
    for threshold in (0.85, 0.90, 0.95):
        groups = {}
        for domain in corr.DOMAIN_ORDER:
            columns = [c for c in cap.DOMAIN_FIELDS[domain] if c in features]
            groups[domain] = corr.connected_groups(
                base[columns].corr(method="spearman"), threshold=threshold
            )
        indices = np.arange(len(df))
        prepared = corr.prepare_raw(df, naval, features, logs, indices)
        domains, _ = corr.fit_corrected_domains(prepared, indices, groups)
        scaled = RobustScaler(quantile_range=(25, 75)).fit_transform(domains)
        pca = PCA(random_state=cap.RANDOM_STATE).fit(scaled)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n90 = int(np.searchsorted(cumulative, 0.90) + 1)
        scores = pca.transform(scaled)[:, :n90]
        labels = KMeans(3, n_init=30, random_state=cap.RANDOM_STATE).fit_predict(scores)
        labels_by_threshold[threshold] = labels
        rows.append({
            "spearman_threshold": threshold,
            "redundancy_groups": sum(len(v) for v in groups.values()),
            "retained_pc90": n90,
            "cumulative_variance": cumulative[n90 - 1],
            "k3_silhouette": silhouette_score(scores, labels),
            "k3_cluster_sizes": "/".join(map(str, sorted(np.bincount(labels)))),
        })

    baseline = labels_by_threshold[0.90]
    for row in rows:
        row["ARI_vs_threshold_0.90"] = adjusted_rand_score(
            baseline, labels_by_threshold[row["spearman_threshold"]]
        )
    output = Path("outputs/capability_profiles/tables/spearman_threshold_sensitivity.csv")
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
