 export PYTHONPATH=$PYTHONPATH:$(pwd)

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER="planTF"
SPLIT=$1
CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"

for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        worker.threads_per_node=24 \
        experiment_uid=$SPLIT/$planner \
        verbose=true \
        planner.imitation_planner.planner_ckpt="/home/jiale/Documents/checkpoints/planTF_official.ckpt"
done


