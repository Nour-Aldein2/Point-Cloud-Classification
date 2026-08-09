train_dataset = PointCloudData(
    root_dir=cfg.data.path,
    num_points=2048,
    split_name="train",
    seed=42,
)

test_dataset = PointCloudData(
    root_dir=cfg.data.path,
    num_points=2048,
    split_name="test",
    seed=42,
)