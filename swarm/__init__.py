"""Seek → Swarm (prototype 3): turn a Seek record into rules a robot swarm can run.

bridge      Seek record → behavior profile (rules in body lengths + cell targets)
sim         the rules engine (no simulator needed)
live_tuner  real-time window: sliders, live charts vs. cell targets, click-to-override
run/render  recorded runs → trajectories, metrics, video
fake_cells  synthetic cells with known rules, for testing
"""
