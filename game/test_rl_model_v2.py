from rl_launcher import main

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    raise SystemExit(main([
        "eval",
        "--run", "runs/26-06-15_12-22-00-155810_PPO_Level5-0",
        "--checkpoint", "best_agent",
        "--enable-window",
    ]))
