# REISD
This is code and data for paper "REISD: Detecting LLM-Generated Text via Iterative Semantic Difference".

# Brief Intro
\begin{table}[H]
    \centering
    \begin{tabular}{@{}lrrrrrr@{}}
        \toprule
        Method & CLTS & XSUM & SQuAD & GovReport & Billsum & HC3 \\
        \midrule
        Log-Likelihood & 55.41 & 55.07 & 48.28 & 36.17 & 34.15 & 51.79 \\
        Rank           & 52.21 & 56.61 & 41.57 & 42.44 & 37.03 & 48.31 \\
        Log-Rank       & 50.37 & 56.42 & 47.64 & 36.93 & 35.79 & 49.98 \\
        Entropy        & 59.35 & 52.31 & 43.78 & 31.54 & 42.64 & 54.64 \\
        Fast-detectGPT & 49.40 & 40.05 & 43.40 & 51.30 & 55.40 & 74.46 \\
        bartscore      & 82.71 & 41.78 & 72.22 & 53.24 & 50.42 & 61.17 \\
        Raidar         & 84.38 & \textbf{97.02} & \textbf{98.58} & \textbf{98.73} & \textbf{96.84} & 59.24 \\
        REISD(Ours)& \textbf{85.93} & 95.23 & 96.58 & 95.34 & 93.23 & \textbf{81.20} \\
        \bottomrule
    \end{tabular}
    \caption{Precision (\%) of each detection method on different datasets}
    \label{tab:precision_results}
\end{table}
