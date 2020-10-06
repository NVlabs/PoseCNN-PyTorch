from render_flex_dataset import PhysicsSeqPoseDataset
from ycb_renderer import *

if __name__ == '__main__':
    dir = sys.argv[1]
    model_dir = sys.argv[2]
    dis_dir = sys.argv[3]
    dataset = PhysicsSeqPoseDataset(dir, model_dir, dis_dir)
    l = (len(dataset))
    print(l)

    renderer = YCBRenderer(256, 256)
    models = [
        "003_cracker_box",
        "002_master_chef_can",
        "011_banana",
    ]
    obj_paths = [
        '{}/models/{}/textured_simple.obj'.format(model_dir, item) for item in models]
    texture_paths = [
        '{}/models/{}/texture_map.png'.format(model_dir, item) for item in models]
    renderer.load_objects(obj_paths, texture_paths)
    renderer.set_fov(40)
    renderer.set_light_pos([0, 0, 0])
    renderer.V = np.eye(4)

    for data_idx in range(20):
        data = dataset[data_idx]
        print(data)
        for i in range(data[0].shape[0]):
            img = data[0][i].transpose(1, 2, 0)
            poses = data[2][i]
            renderer.set_poses(poses)
            frame = renderer.render()
            cv2.imshow('test', cv2.cvtColor(np.concatenate([img] + [item[:, :, :3] for item in frame], axis=1),
                                            cv2.COLOR_RGB2BGR))
            q = cv2.waitKey(100)

    renderer.release()
