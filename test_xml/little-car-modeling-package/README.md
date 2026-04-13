# Little Car Modeling Package

这是一个可直接给同学使用的小车模型包，包含：

- **URDF**：`assets/urdf/little_car.urdf`
- **MuJoCo 场景 / MJCF**：
  - `assets/mjcf/little_car.xml`
  - `assets/mjcf/little_car_scene.xml`
- **当前版本 3D 模型**：
  - `assets/meshes/chassis_top.obj`
  - `assets/meshes/wheel.obj`
- **参数化生成脚本源码**：
  - `car_model/parameters.py`
  - `car_model/mesh_utils.py`
  - `car_model/generate_model.py`
  - `car_model/square_eval.py`
- **示例脚本**：
  - `scripts/smoke_test_mujoco.py`
  - `scripts/drive_smoke_test.py`
  - `scripts/square_path_eval.py`
  - `scripts/render_square_path_video.py`

> 这个压缩包**不包含视频和分析产物**，只包含建模和使用所需的原始文件。

---

## 1. 模型简介

这个小车是一个用于 MuJoCo / URDF 实验的小型差速两轮车模型。

当前版本特点：

- 车体俯视为方形，默认边长 `0.07 m`
- 车体总高度（顶部到地面）约 `0.08 m`
- 两侧轮子直径 `0.05 m`
- 后部有一个**裁剪在车体内部**的支撑块，用于形成三点支撑
- 车体质量分布后偏，后半部分密度高于前半部分，以提高稳定性
- 默认前进方向为 **`+x`**
- 后支撑位于车尾（负 `x` 方向）

---

## 2. 目录说明

```text
assets/
  meshes/
    chassis_top.obj
    wheel.obj
  mjcf/
    little_car.xml
    little_car_scene.xml
  urdf/
    little_car.urdf

car_model/
  __init__.py
  parameters.py
  mesh_utils.py
  generate_model.py
  square_eval.py

scripts/
  smoke_test_mujoco.py
  drive_smoke_test.py
  square_path_eval.py
  render_square_path_video.py
```

---

## 3. 环境要求

推荐使用当前项目开发时使用的 Python：

```bash
/home/shanoriel/miniforge3/envs/leworldmodel/bin/python
```

如果你自己配置环境，至少需要：

- Python 3.10+
- `mujoco`
- `numpy`
- `imageio`
- `imageio-ffmpeg`

如果要离屏渲染视频，通常还需要：

- EGL / headless OpenGL 支持

---

## 4. 如何重新生成模型

如果修改了参数文件：

- `car_model/parameters.py`

就可以重新生成当前版本的小车模型：

```bash
cd little-car-modeling
/home/shanoriel/miniforge3/envs/leworldmodel/bin/python -m car_model.generate_model
```

生成结果会写到：

- `assets/meshes/`
- `assets/mjcf/`
- `assets/urdf/`

---

## 5. 如何使用 URDF

URDF 文件在：

```text
assets/urdf/little_car.urdf
```

注意：

- 这是最重要的交换格式文件
- MuJoCo 直接导入 URDF 时，固定连接的部分可能会被折叠，因此如果要做稳定性和驱动验证，**更推荐直接使用 MJCF 版本**

---

## 6. 如何使用 MuJoCo 场景

### 小车 MJCF 模型

```text
assets/mjcf/little_car.xml
```

### 带地面和相机的场景

```text
assets/mjcf/little_car_scene.xml
```

如果你想直接在 MuJoCo 里加载场景，优先用：

```text
assets/mjcf/little_car_scene.xml
```

因为它已经包含：

- gravity
- 地面
- 相机
- 小车模型 include

---

## 7. 如何做快速验证

### 7.1 基础加载验证

```bash
cd little-car-modeling
/home/shanoriel/miniforge3/envs/leworldmodel/bin/python scripts/smoke_test_mujoco.py
```

### 7.2 基本驱动验证

```bash
/home/shanoriel/miniforge3/envs/leworldmodel/bin/python scripts/drive_smoke_test.py
```

### 7.3 方形轨迹验证

```bash
/home/shanoriel/miniforge3/envs/leworldmodel/bin/python scripts/square_path_eval.py
```

---

## 8. 如何渲染视频

```bash
cd little-car-modeling
MUJOCO_GL=egl /home/shanoriel/miniforge3/envs/leworldmodel/bin/python scripts/render_square_path_video.py
```

默认会生成方形轨迹视频和相关产物。

---

## 9. 最重要的可调参数

主要在：

```text
car_model/parameters.py
```

重点可以改的包括：

- `body_size_m`
- `body_height_m`
- `wheel_diameter_m`
- `support_offset_x_m`
- `support_length_m`
- `support_width_m`
- `support_mass_kg`
- `drive_force_limit_n_m`

如果改了参数，记得重新运行：

```bash
/home/shanoriel/miniforge3/envs/leworldmodel/bin/python -m car_model.generate_model
```

---

## 10. 当前最终版本说明

当前最终版本采用了这些关键设计：

- 后支撑块已经**裁剪在车体 footprint 内部**
- 后支撑质量使用较轻的版本，以保持更好的运动速度
- 后支撑位置相对更靠后，以降低转弯时的几何扰动
- 驱动力矩上限经过调节，使方形轨迹更规整

---

## 11. 建议给同学的使用方式

如果只是要复现当前结果：

1. 解压缩包
2. 进入 `little-car-modeling`
3. 先运行：

```bash
/home/shanoriel/miniforge3/envs/leworldmodel/bin/python scripts/smoke_test_mujoco.py
```

4. 再运行：

```bash
/home/shanoriel/miniforge3/envs/leworldmodel/bin/python scripts/square_path_eval.py
```

5. 如果需要视频：

```bash
MUJOCO_GL=egl /home/shanoriel/miniforge3/envs/leworldmodel/bin/python scripts/render_square_path_video.py
```

如果只是想把模型导入别的系统，优先看：

- `assets/urdf/little_car.urdf`
- `assets/mjcf/little_car.xml`
- `assets/mjcf/little_car_scene.xml`

---

## 12. 联系使用建议

如果后续要继续改：

- 先改 `parameters.py`
- 再重新生成模型
- 再跑 smoke test / square eval

这样最稳，不容易把模型和脚本状态搞乱。
