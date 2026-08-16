from rl_launcher import main

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    raise SystemExit(main([
        "eval",
        "--run", "runs/SKRL/26-06-15_12-22-00-155810_SKRL_PPO_flat_walk",
        "--checkpoint", "best_agent",
        "--enable-window",
    ]))
