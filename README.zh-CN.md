# Depth Surge 3D

Depth Surge 3D 使用单目深度估计把普通 2D 视频转换为左右眼 3D VR 视频。

## Windows 安装与首次转换

推荐 Windows x64 和 64 位 CPython 3.11，缺少 Python 时 uv 可以下载。先安装 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、[FFmpeg](https://ffmpeg.org/download.html) 和 [微软 C++ 生成工具](https://visualstudio.microsoft.com/visual-cpp-build-tools/)。在生成工具中勾选「使用 C++ 的桌面开发」和 Windows SDK。重新打开 PowerShell，确认 `ffmpeg -version` 和 `ffprobe -version` 都能运行。

```powershell
git clone https://github.com/sdbds/depth-surge-3d.git
cd depth-surge-3d
.\1.install-uv-qinglong.ps1
.\.venv\Scripts\depth-surge-3d.exe --help
```

脚本按 `uv.lock` 安装项目本身，编译并检查 Quality 扩展，不下载模型权重。已有 `.venv` 不兼容时不会删除它；用 `-EnvironmentPath D:\envs\depth-surge-311` 可另建环境。需要 MoGe-2 才加 `-WithMoGe2`。

首次转换可用 V3 Small，不需要另克隆模型源码。把 `input.mp4` 换成自己的视频文件；第一次运行要联网获取权重。此命令只转换第一秒：

```powershell
.\.venv\Scripts\depth-surge-3d.exe input.mp4 --depth-model-version v3 --model-size vits --end 00:01 --vr-resolution custom:640x360 --stereo-render-mode quality
```

要用网页界面，在仓库目录运行 `.\.venv\Scripts\python.exe app.py`，再打开 [localhost:5000](http://localhost:5000)。CLI 不指定后端会选 V2；它需要独立源码和对应权重。其他平台、Linux 可选 FlashAttention、V2 准备步骤和离线权重校验见[安装指南](docs/INSTALLATION.md)。

## 立体渲染模式

- `quality`：离线最高画质。使用编译的 C++17 原生扩展与 RGB 边缘约束重建深度轮廓，
  支持 1080p 与 4K 渲染，并保留细发丝、羽毛、枪管与缎带等结构；补洞仅作用于遮挡显露区。
- `fast`：快速路径。基于 GPU Triton 内核，用于快速预览、迭代与低算力设备。

新任务未显式指定模式时，CUDA 设备默认选择 `quality`，CPU 默认选择 `fast`。恢复旧任务时
保留已保存模式；schema 1 到 4 的历史任务迁移为 `fast` 和 8 像素上限。

```powershell
# 离线最高画质，1080p 基准补洞上限 8 像素
.\.venv\Scripts\depth-surge-3d.exe input.mp4 --depth-model-version v3 --model-size vits --stereo-render-mode quality --occlusion-fill-max-px 8

# 快速模式
.\.venv\Scripts\depth-surge-3d.exe input.mp4 --depth-model-version v3 --model-size vits --stereo-render-mode fast
```

`--occlusion-fill-max-px` 的合法范围为 `1..32`，会随输出高度缩放，建议先使用 8。
Quality 也可在 CPU 上运行，但仅适合离线慢速处理。

## 诊断与恢复

立体渲染阶段的汇总覆盖与修复指标以低内存流式聚合写入
`04_stereo_diagnostics/summary.json`。断点续跑通过阶段元数据与帧清单校验复用已完成阶段；
成片输出支持 `hevc_nvenc`、`libx265`、`libx264` 零拷贝流式直出，并以轻量 `ffprobe` 校验封装与音画同步。

安装、Web UI 和完整参数请参阅 [英文 README](README.md)、
[用户指南](docs/USER_GUIDE.md) 与 [参数参考](docs/PARAMETERS.md)。
