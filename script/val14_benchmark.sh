 export PYTHONPATH=$PYTHONPATH:$(pwd)

# cwd=$(pwd)
CKPT_ROOT="~/Documents/checkpoints/"

PLANNER="planTF"
CHALLENGES="closed_loop_reactive_agents"

for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan \
        scenario_filter=val14 \
        worker.threads_per_node=30 \
        experiment_uid=val14/$planner \
        verbose=true \
        planner.imitation_planner.planner_ckpt="/home/jiale/Documents/checkpoints/planTF_official.ckpt"
done


# sh ./script/val14_benchmark.sh
