python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=16 \
  scenario_builder=nuplan cache.cache_path=/home/jiale/Documents/exp/cache_new cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=64 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=35 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.check_val_every_n_epoch=5 \
  wandb.mode=online wandb.project=nuplan wandb.name=ablation_waytpoints_num \
  model.waypoints_number = 10



python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=16 \
  scenario_builder=nuplan cache.cache_path=/home/jiale/Documents/exp/cache_new cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=64 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=35 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.check_val_every_n_epoch=5 \
  wandb.mode=online wandb.project=nuplan wandb.name=ablation_waytpoints_num \
  model.waypoints_number = 30



python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=16 \
  scenario_builder=nuplan cache.cache_path=/home/jiale/Documents/exp/cache_new cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=64 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=35 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.check_val_every_n_epoch=5 \
  wandb.mode=online wandb.project=nuplan wandb.name=ablation_waytpoints_num \
  model.waypoints_number = 40