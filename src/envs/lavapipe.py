"""Select Mesa's CPU Vulkan driver (lavapipe) as ManiSkill3's render device.

CHTC containers get CUDA but not the NVIDIA Vulkan driver, so SAPIEN cannot use the GPU for rendering.
SAPIEN's "cpu" device never renders; lavapipe must be selected as a Vulkan physical device by PCI address.
ManiSkill splits backend names on ":", so the PCI address is registered under the alias "lavapipe".
Requires Mesa >= 24.3 and VK_ICD_FILENAMES / VK_DRIVER_FILES pointing at the lavapipe ICD (set in the container).
"""
LAVAPIPE_BACKEND = "lavapipe"
LAVAPIPE_PCI = "pci:0000:00:00.0"


def register_lavapipe() -> str:
	from mani_skill.envs.utils.system import backend

	backend.render_backend_name_mapping[LAVAPIPE_BACKEND] = LAVAPIPE_PCI
	return LAVAPIPE_BACKEND
