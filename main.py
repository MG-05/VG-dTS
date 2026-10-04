from src.adts.experiments import (
    reproduce_figure1,
    reproduce_rigorous_env_suite,
    run_paper_algorithm_benchmarks,
    run_rigorous_env_suite_oracle_experiment,
    run_single_env_lambda_tuning_experiment,
    run_slow_varying_sinusoid_oracle_experiment,
)


def main() -> None:
    # oracle_plot_path = run_slow_varying_sinusoid_oracle_experiment(show=False)
    # print(f"Saved dynamic-oracle plot to: {oracle_plot_path}")

    # rigorous_plot_path = run_rigorous_env_suite_oracle_experiment(show=False)
    # print(f"Saved rigorous environment suite plot to: {rigorous_plot_path}")

    rigorous_reward_regret_path = reproduce_rigorous_env_suite(n_runs=100, seed=0, show=False)
    print(f"Saved rigorous reward/regret plot to: {rigorous_reward_regret_path}")

    figure1_plot_path = reproduce_figure1(n_runs=100, seed=0, show=False)
    print(f"Saved figure 1 reproduction to: {figure1_plot_path}")

    tuning_plot_path, tuned_results = run_single_env_lambda_tuning_experiment(
        environment="fast",
        n_runs=100,
        tuning_runs=50,
        seed=0,
        show=False,
    )
    print(f"Saved single-environment tuning plot to: {tuning_plot_path}")
    print(
        "Best tuned policies (final normalized regret): "
        + ", ".join(
            f"{name}={result.final_normalized_regret:.4f}"
            for name, result in sorted(tuned_results.items())
        )
    )

    nonstationary_plot_path, recovering_plot_path, summary_path, _ = run_paper_algorithm_benchmarks(
        n_runs_nonstationary=20,
        n_runs_recovering=20,
        seed=0,
        output_dir="report",
        make_plots=True,
    )
    print(f"Saved paper non-stationary heatmap to: {nonstationary_plot_path}")
    print(f"Saved paper recovering benchmark plot to: {recovering_plot_path}")
    print(f"Saved paper benchmark summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
