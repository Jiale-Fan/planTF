 export PYTHONPATH=$PYTHONPATH:$(pwd)

# cwd=$(pwd)
CKPT_ROOT="~/Documents/checkpoints/"

PLANNER="planTF"
CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"

for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan \
        scenario_filter=val14 \
        worker.threads_per_node=30 \
        experiment_uid=val14/$planner \
        verbose=true \
        planner.imitation_planner.planner_ckpt="/home/jiale/Documents/checkpoints/2211_1.4_nopre.ckpt"
done


# sh ./script/plantf_benchmarks.sh test14-random
# sh ./script/plantf_benchmarks.sh test14-hard