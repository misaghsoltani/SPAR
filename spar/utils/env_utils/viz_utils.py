"""Visualization utilities for the SPAR framework."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
import warnings

from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from numpy import float32, uint8

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypedDict

    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from typing_extensions import Unpack

    class _InteractiveCubeAxesKwargs(TypedDict, total=False):
        aspect: str
        xlim: tuple[float, float]
        ylim: tuple[float, float]
        frameon: bool
        xticks: list[float]
        yticks: list[float]


class Quaternion:
    """Quaternion operations used by the Cube3 renderer."""

    def __init__(self, x: NDArray[float32] | list[float] | NDArray[np.float64]) -> None:
        """Initializes the Quaternion.

        Args:
            x: The quaternion components.
        """
        self.x: NDArray[float32] = np.asarray(x, dtype=float32)

    @classmethod
    def from_v_theta(
        cls, v: NDArray[float32] | list[float] | tuple[float, ...], theta: NDArray[float32] | float
    ) -> Quaternion:
        """Construct quaternions from unit vectors v and rotation angles theta.

        Args:
            v: Array of vectors, last dimension 3. Vectors will be normalized.
            theta: Array of rotation angles in radians, shape = v.shape[:-1].

        Returns:
            Quaternion representing the rotations.
        """
        theta_arr: NDArray[float32] = np.asarray(theta, dtype=float32)
        v_arr: NDArray[float32] = np.asarray(v, dtype=float32)
        s: NDArray[float32] = np.sin(float32(0.5) * theta_arr).astype(float32)
        c: NDArray[float32] = np.cos(float32(0.5) * theta_arr).astype(float32)

        v_arr = np.asarray(v_arr * s / np.sqrt(np.sum(v_arr * v_arr, -1)), dtype=float32)
        x_shape: tuple[int, ...] = (*v_arr.shape[:-1], 4)

        x: NDArray[float32] = np.ones(x_shape, dtype=float32).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v_arr.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __repr__(self) -> str:
        """Returns a string representation of the Quaternion.

        Returns:
            str: String representation of the Quaternion.
        """
        return f"Quaternion:\n{self.x.__repr__()}"

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Multiplies two quaternions.

        Args:
            other (Quaternion): The other quaternion to multiply with.

        Returns:
            Quaternion: The product of the two quaternions.
        """
        sxr: NDArray[float32] = self.x.reshape((*self.x.shape[:-1], 4, 1))
        oxr: NDArray[float32] = other.x.reshape((*other.x.shape[:-1], 1, 4))

        prod: NDArray[float32] = sxr * oxr
        return_shape: tuple[int, ...] = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret: NDArray[float32] = np.array(
            [
                (prod[0, 0] - prod[1, 1] - prod[2, 2] - prod[3, 3]),
                (prod[0, 1] + prod[1, 0] + prod[2, 3] - prod[3, 2]),
                (prod[0, 2] - prod[1, 3] + prod[2, 0] + prod[3, 1]),
                (prod[0, 3] + prod[1, 2] - prod[2, 1] + prod[3, 0]),
            ],
            dtype=float32,
            order="F",
        ).T
        return self.__class__(ret.reshape(return_shape))

    def as_v_theta(self) -> tuple[NDArray[float32], NDArray[float32]]:
        """Returns the v, theta equivalent of the (normalized) quaternion.

        Returns:
            The unit vector and rotation angle.
        """
        x: NDArray[float32] = self.x.reshape((-1, 4)).T

        # compute theta
        norm: NDArray[float32] = np.sqrt((x**2).sum(0))
        theta: NDArray[float32] = 2 * np.arccos(x[0] / norm)

        # compute the unit vector
        v: NDArray[float32] = np.array(x[1:], order="F", copy=True)
        v /= np.sqrt(np.sum(v**2, 0))

        # reshape the results
        v = v.T.reshape((*self.x.shape[:-1], 3))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    def as_rotation_matrix(self) -> NDArray[float32]:
        """Returns the rotation matrix of the (normalized) quaternion.

        Returns:
            The rotation matrix.
        """
        theta: NDArray[float32]
        v: NDArray[float32]

        v, theta = self.as_v_theta()

        shape: tuple[int, ...] = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c: NDArray[float32] = np.cos(theta)
        s: NDArray[float32] = np.sin(theta)

        mat: NDArray[float32] = np.array(
            [
                [v[0] * v[0] * (1.0 - c) + c, v[0] * v[1] * (1.0 - c) - v[2] * s, v[0] * v[2] * (1.0 - c) + v[1] * s],
                [v[1] * v[0] * (1.0 - c) + v[2] * s, v[1] * v[1] * (1.0 - c) + c, v[1] * v[2] * (1.0 - c) - v[0] * s],
                [v[2] * v[0] * (1.0 - c) - v[1] * s, v[2] * v[1] * (1.0 - c) + v[0] * s, v[2] * v[2] * (1.0 - c) + c],
            ],
            order="F",
        ).T
        return mat.reshape((*shape, 3, 3))

    def rotate(self, points: NDArray[float32]) -> NDArray[float32]:
        """Rotates the given points using the quaternion.

        Args:
            points: The points to rotate.

        Returns:
            The rotated points.
        """
        rot_mat: NDArray[float32] = self.as_rotation_matrix()
        return np.dot(points, rot_mat.T)


def project_points(
    points: NDArray[float32], q: Quaternion, view: NDArray[float32], vertical: NDArray[float32] | None
) -> NDArray[float32]:
    """Project points using a quaternion q and a view v.

    Args:
        points: Array of last-dimension 3.
        q: Quaternion representation of the rotation.
        view: Length-3 vector giving the point of view.
        vertical: Direction of y-axis for view. An error will be raised if it is
            parallel to the view.

    Returns:
        Array of projected points: same shape as points.
    """
    if vertical is None:
        vertical = np.array([0, 1, 0], dtype=float32)
    points = np.asarray(points, dtype=float32)
    view = np.asarray(view, dtype=float32)

    xdir: NDArray[float32] = np.cross(vertical, view).astype(float32)

    if np.all(xdir == 0):
        raise ValueError("vertical is parallel to v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # get the unit vector corresponding to vertical
    ydir: NDArray[float32] = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    # normalize the viewer location: this is the z-axis
    v2: float32 = np.dot(view, view)
    zdir: NDArray[float32] = view / np.sqrt(v2)

    # rotate the points
    rot_mat: NDArray[float32] = q.as_rotation_matrix()
    r_pts: NDArray[float32] = np.dot(points, rot_mat.T)

    # project the points onto the view
    dpoint: NDArray[float32] = r_pts - view
    dpoint_view: NDArray[float32] = np.dot(dpoint, view).reshape((*dpoint.shape[:-1], 1))
    dproj: NDArray[float32] = -dpoint * v2 / dpoint_view

    trans: list[int] = [*list(range(1, dproj.ndim)), 0]
    return np.array([np.dot(dproj, xdir), np.dot(dproj, ydir), -np.dot(dpoint, zdir)]).transpose(trans)


def euler_to_quaternion(yaw: float, pitch: float, roll: float) -> Quaternion:
    """Convert Euler angles to a quaternion.

    The Euler angles are interpreted as relative offsets from a default (clean) view.

    Args:
        yaw (float): Rotation around the z-axis in degrees.
        pitch (float): Rotation around the y-axis in degrees.
        roll (float): Rotation around the x-axis in degrees.

    Returns:
        Quaternion: The quaternion representing the combined rotation.

    """
    # Convert degrees to radians
    yaw_r: float = np.deg2rad(yaw)
    pitch_r: float = np.deg2rad(pitch)
    roll_r: float = np.deg2rad(roll)

    # Trigonometric values
    cy: float = np.cos(yaw_r * 0.5)
    sy: float = np.sin(yaw_r * 0.5)
    cp: float = np.cos(pitch_r * 0.5)
    sp: float = np.sin(pitch_r * 0.5)
    cr: float = np.cos(roll_r * 0.5)
    sr: float = np.sin(roll_r * 0.5)

    # Quaternion components
    w: float = cr * cp * cy + sr * sp * sy
    x: float = sr * cp * cy - cr * sp * sy
    y: float = cr * sp * cy + sr * cp * sy
    z: float = cr * cp * sy - sr * sp * cy

    return Quaternion(np.array([w, x, y, z], dtype=float32))


# ----------------------------------------------
# Adapted from Matplotlib Rubik's cube simulator
# written by Jake Vanderplas, which is adapted
# from cube code written by David Hogg
#   https://github.com/davidwhogg/MagicCube
# ----------------------------------------------


class InteractiveCube(Axes):
    """InteractiveCube is a matplotlib Axes subclass for visualizing and interacting with a cube.

    Attributes:
        base_face (NDArray): Base face coordinates.
        stickerwidth (float): Width of a sticker.
        stickermargin (float): Margin for sticker placement.
        stickerthickness (float): Thickness of a sticker.
        base_sticker (NDArray): Base sticker coordinates.
        base_face_centroid (NDArray): Base centroid for a face.
        base_sticker_centroid (NDArray): Base centroid for a sticker.
        defer_draws (bool): When True, state updates skip canvas draw
            scheduling. Batch renderers set this because they call an explicit
            ``canvas.draw()`` before reading the pixel buffer, so intermediate
            draws are discarded work. On the headless Agg backend
            ``draw_idle()`` draws eagerly, which makes those intermediate
            draws the dominant rendering cost.

    """

    defer_draws: bool = False

    # Pre-compute and store constants as class attributes for better performance
    base_face: NDArray[float32] = np.ascontiguousarray(
        [[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=float32
    )
    stickerwidth: float = 0.9
    stickermargin: float = 0.5 * (1.0 - stickerwidth)
    stickerthickness: float = 0.001
    d1: float
    d2: float
    d3: float
    d1, d2, d3 = (1 - stickermargin, 1 - 2 * stickermargin, 1 + stickerthickness)
    base_sticker: NDArray[float32] = np.ascontiguousarray(
        [
            [d1, d2, d3],
            [d2, d1, d3],
            [-d2, d1, d3],
            [-d1, d2, d3],
            [-d1, -d2, d3],
            [-d2, -d1, d3],
            [d2, -d1, d3],
            [d1, -d2, d3],
            [d1, d2, d3],
        ],
        dtype=float32,
    )
    base_face_centroid: NDArray[float32] = np.ascontiguousarray([[0, 0, 1]], dtype=float32)
    base_sticker_centroid: NDArray[float32] = np.ascontiguousarray([[0, 0, 1 + stickerthickness]], dtype=float32)
    # Pre-defined view and camera orientations
    vertical_axis: NDArray[float32] = np.ascontiguousarray([0, 1, 0], dtype=float32)

    def __init__(
        self,
        n: int,
        colors: NDArray[uint8],
        view: Sequence[float] = (0, 0, 10),
        fig: Figure | None = None,
        **kwargs: Unpack[_InteractiveCubeAxesKwargs],
    ) -> None:
        """Initialize an InteractiveCube instance.

        Args:
            n: Number of subdivisions per face.
            colors: Color assignments for stickers.
            view: Camera view vector. Defaults to (0, 0, 10).
            fig: Matplotlib figure. Defaults to current figure.
            **kwargs: Additional keyword arguments for the Axes.
        """
        self.colors: NDArray[uint8] = np.ascontiguousarray(colors, dtype=uint8)

        # Suppress warning about moveable paths - this is handled by matplotlib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            eye = np.eye(3, dtype=np.float64)
            # Precompute rotations for cube faces
            self.rots: list[Quaternion] = [Quaternion.from_v_theta(eye[0], theta) for theta in (np.pi / 2, -np.pi / 2)]
            self.rots += [Quaternion.from_v_theta(eye[1], theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

            # Pre-compute rotation matrices for faster initialization
            self.rot_matrices: NDArray[float32] = np.array(
                [rot.as_rotation_matrix() for rot in self.rots], dtype=float32
            )

        self._move_list: list[str] = []
        self.N: int = n
        # self._prevStates: list[NDArray[np.int32]] = []
        self._view: NDArray[float32] = np.ascontiguousarray(view, dtype=float32)
        self._start_rot: Quaternion = Quaternion.from_v_theta((1, -1, 0), -np.pi / 6)
        self._grey_stickers: list[int] = []
        self._black_stickers: list[int] = []

        if fig is None:
            fig = plt.gcf()
        # Disable default key press events
        callbacks = fig.canvas.callbacks.callbacks
        callbacks.pop("key_press_event", None)

        rect: tuple[float, float, float, float] = (0, 0.16, 1, 0.84)
        xlim: tuple[float, float] = kwargs.setdefault("xlim", (-1.7, 1.5))
        ylim: tuple[float, float] = kwargs.setdefault("ylim", (-1.5, 1.7))
        kwargs.setdefault("aspect", "equal")
        kwargs.setdefault("frameon", False)
        kwargs.setdefault("xticks", [])
        kwargs.setdefault("yticks", [])
        super().__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(NullFormatter())
        self.yaxis.set_major_formatter(NullFormatter())

        self._start_xlim: tuple[float, float] = xlim
        self._start_ylim: tuple[float, float] = ylim

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD: tuple[float, float, float] = (1, 0, 0)
        self._step_UD: float = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR: tuple[float, float, float] = (0, -1, 0)
        self._step_LR: float = 0.01

        self._ax_LR_alt: tuple[float, float, float] = (0, 0, 1)

        self._current_rot: Quaternion = self._start_rot
        self._face_polys: list[Polygon] | None = None
        self._sticker_polys: list[Polygon] | None = None

        # Cache these values for reuse
        self._proj_stickers: NDArray[float32] | None = None
        self._proj_faces: NDArray[float32] | None = None
        self._proj_face_centroids: NDArray[float32] | None = None
        self._proj_sticker_centroids: NDArray[float32] | None = None
        self._stickers_2d: NDArray[float32] | None = None
        self._faces_2d: NDArray[float32] | None = None
        self._face_zorders: NDArray[float32] | None = None
        self._sticker_zorders: NDArray[float32] | None = None

        self.plastic_color: str = "black"

        # Camera offset storage for effects system
        self._camera_offsets: dict[str, float] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        self.face_colors: list[str] = ["w", "#ffcf00", "#ff6f00", "#cf0000", "#00008f", "#009f0f", "gray", "none"]

        self._initialize_arrays()
        self._draw_cube()

    def set_rot(self, rot: int) -> None:
        """Set the cube rotation to one of two predefined rotations.

        Args:
            rot (int): Rotation index.
        """
        if rot == 0:
            self._current_rot = Quaternion.from_v_theta((-0.53180525, 0.83020462, 0.16716299), 0.95063829)
        elif rot == 1:
            self._current_rot = Quaternion.from_v_theta((0.9248325, 0.14011997, -0.35362584), 2.49351394)

        # Invalidate cached projections
        self._invalidate_projections()
        self._draw_cube()

    def set_camera_pose(self, yaw: float, pitch: float, roll: float, base_quat: Quaternion) -> None:
        """Set an arbitrary camera pose.

        Applies a relative rotation (in Euler angles in degrees) to the provided base quaternion.

        Args:
            yaw (float): Rotation around the z-axis in degrees.

            pitch (float): Rotation around the y-axis in degrees.

            roll (float): Rotation around the x-axis in degrees.

            base_quat (Quaternion): Base quaternion for the clean view.
        """
        q_offset: Quaternion = euler_to_quaternion(yaw, pitch, roll)
        self._current_rot = q_offset * base_quat

        # Invalidate cached projections so they get recomputed
        self._invalidate_projections()
        self._draw_cube()

    def apply_camera_offsets(self) -> None:
        """Apply stored camera offsets to the current rotation.

        This method should be called after set_rot() to apply any camera effects.
        """
        if hasattr(self, "_camera_offsets") and any(self._camera_offsets.values()):
            yaw: float = self._camera_offsets.get("yaw", 0.0)
            pitch: float = self._camera_offsets.get("pitch", 0.0)
            roll: float = self._camera_offsets.get("roll", 0.0)

            if yaw or pitch or roll:
                # Get the current rotation after set_rot
                base_quat: Quaternion = self._current_rot
                # Apply offset relative to the current rotation
                q_offset: Quaternion = euler_to_quaternion(yaw, pitch, roll)
                self._current_rot = q_offset * base_quat

                # Invalidate cached projections so they get recomputed
                self._invalidate_projections()
                # Update geometry only (position/z-order) without resetting colors
                self._update_geometry_only()

    def set_camera_offsets(self, yaw: float, pitch: float, roll: float) -> None:
        """Store camera offsets to apply after `set_rot`.

        Args:
            yaw: Rotation around z-axis in degrees.
            pitch: Rotation around y-axis in degrees.
            roll: Rotation around x-axis in degrees.
        """
        self._camera_offsets["yaw"] = yaw
        self._camera_offsets["pitch"] = pitch
        self._camera_offsets["roll"] = roll

    def get_start_rotation(self) -> Quaternion:
        """Return the canonical starting rotation used by presets."""
        return self._start_rot

    def get_rotation_with_camera_offsets(self) -> Quaternion:
        """Return current rotation with pending camera offsets applied."""
        current_rot: Quaternion = self._current_rot
        if any(self._camera_offsets.values()):
            yaw: float = self._camera_offsets.get("yaw", 0.0)
            pitch: float = self._camera_offsets.get("pitch", 0.0)
            roll: float = self._camera_offsets.get("roll", 0.0)
            if yaw or pitch or roll:
                q_offset: Quaternion = euler_to_quaternion(yaw, pitch, roll)
                return q_offset * current_rot
        return current_rot

    def set_grey_stickers(self, sticker_indices: Sequence[int]) -> None:
        """Set sticker indices rendered gray on the next redraw.

        Args:
            sticker_indices: Sticker indices to render gray.
        """
        self._grey_stickers = list(sticker_indices)

    def redraw(self) -> None:
        """Redraw cube geometry/colors using current state."""
        self._draw_cube()

    def ensure_sticker_polygons(self) -> list[Polygon]:
        """Return sticker polygons, drawing the cube if needed."""
        if self._sticker_polys is None:
            self._draw_cube()
        if self._sticker_polys is None:
            raise RuntimeError("Sticker polygons are not available after drawing the cube")
        return self._sticker_polys

    def zoom_geometry(self, zoom_factor: float, offset_x: float = 0.0, offset_y: float = 0.0) -> None:
        """Apply zoom and positional offsets to cube geometry.

        Args:
            zoom_factor: Multiplicative geometry scale.
            offset_x: Horizontal offset in world coordinates.
            offset_y: Vertical offset in world coordinates.
        """
        self._faces *= zoom_factor
        self._stickers *= zoom_factor
        self._sticker_centroids *= zoom_factor
        np.multiply(self._face_centroids[:, :3], zoom_factor, out=self._face_centroids[:, :3])

        if abs(offset_x) > 1e-6 or abs(offset_y) > 1e-6:
            offset_3d: NDArray[float32] = np.array([offset_x, offset_y, 0.0], dtype=float32)
            self._faces += offset_3d
            self._stickers += offset_3d
            self._sticker_centroids += offset_3d
            np.add(self._face_centroids[:, :3], offset_3d, out=self._face_centroids[:, :3])

        with contextlib.suppress(Exception):
            half_width: float = (self._start_xlim[1] - self._start_xlim[0]) / (2 * zoom_factor)
            half_height: float = (self._start_ylim[1] - self._start_ylim[0]) / (2 * zoom_factor)
            x_center: float = (self._start_xlim[0] + self._start_xlim[1]) / 2 + offset_x
            y_center: float = (self._start_ylim[0] + self._start_ylim[1]) / 2 + offset_y
            self.set_xlim((x_center - half_width, x_center + half_width))
            self.set_ylim((y_center - half_height, y_center + half_height))

        with contextlib.suppress(Exception):
            self.margins(x=0, y=0)
            self.set_position((0, 0, 1, 1))

        self._draw_cube()

    def _update_geometry_only(self) -> None:
        """Update polygon geometry (positions, z-order) without changing colors.

        This is used when camera rotation changes but we want to preserve
        colors set by effects like lighting.
        """
        if self._sticker_polys is None or self._face_polys is None:
            # No polygons exist yet, need full draw
            self._draw_cube()
            return

        # Compute projections if not already cached
        self._compute_projections()

        # Compute projections before reading face coordinates.
        assert self._stickers_2d is not None
        assert self._faces_2d is not None
        assert self._face_zorders is not None
        assert self._sticker_zorders is not None

        # Update existing polygons geometry only
        for i in range(len(self._sticker_polys)):
            self._face_polys[i].set_xy(self._faces_2d[i])
            self._face_polys[i].set_zorder(self._face_zorders[i])

            self._sticker_polys[i].set_xy(self._stickers_2d[i])
            self._sticker_polys[i].set_zorder(self._sticker_zorders[i])

        # Defer the draw until the GUI event loop is idle. Batch renderers draw explicitly.
        if not self.defer_draws:
            self.figure.canvas.draw_idle()

    def _invalidate_projections(self) -> None:
        """Reset cached projection data."""
        self._proj_stickers = None
        self._proj_faces = None
        self._proj_face_centroids = None
        self._proj_sticker_centroids = None
        self._stickers_2d = None
        self._faces_2d = None
        self._face_zorders = None
        self._sticker_zorders = None

    def _initialize_arrays(self) -> None:
        """Initialize arrays for centroids, faces, and stickers for the cube."""
        # 1) compute the grid of translations (p = N*N)
        cubie_width: float = 2.0 / self.N
        coords: NDArray[float32] = np.linspace(-1 + cubie_width / 2, 1 - cubie_width / 2, self.N, dtype=float32)
        gx: NDArray[float32]
        gy: NDArray[float32]
        gx, gy = np.meshgrid(coords, coords, indexing="ij")
        # translations: (p,1,3)
        translations: NDArray[float32] = np.stack((gx, gy, np.zeros_like(gx)), axis=-1).reshape(-1, 1, 3)
        p: int = translations.shape[0]  # = N*N

        # 2) scale the base shapes
        factor: NDArray[float32] = np.array([1.0 / self.N, 1.0 / self.N, 1.0], dtype=float32)
        base_face_scaled: NDArray[float32] = factor * self.base_face  # (F,3)
        base_sticker_scaled: NDArray[float32] = factor * self.base_sticker  # (S,3)

        # 3) build raw (unrotated) arrays by broadcasting
        #    faces_base:    (p,F,3)
        #    stickers_base: (p,S,3)
        faces_base: NDArray[float32] = translations + base_face_scaled
        stickers_base: NDArray[float32] = translations + base_sticker_scaled

        #    centroid bases: (p,3)
        face_centroid_base: NDArray[float32] = translations[:, 0, :] + self.base_face_centroid
        sticker_centroid_base: NDArray[float32] = translations[:, 0, :] + self.base_sticker_centroid

        # 4) grab all 6 rotation matrices in one go: (6,3,3)
        rot_mats: NDArray[float32] = np.stack([r.as_rotation_matrix() for r in self.rots], axis=0)

        # 5) apply rotations in bulk via einsum
        #    faces_t:    (6,p,F,3)
        #    stickers_t: (6,p,S,3)
        #    face_ctr_t: (6,p,3)
        #    stickr_ctr_t:(6,p,3)
        faces_t: NDArray[float32] = np.einsum("fij,npj->fnpi", rot_mats, faces_base)
        stickers_t: NDArray[float32] = np.einsum("fij,npj->fnpi", rot_mats, stickers_base)
        face_ctr_t: NDArray[float32] = np.einsum("fij,nj->fni", rot_mats, face_centroid_base)
        stickr_ctr_t: NDArray[float32] = np.einsum("fij,nj->fni", rot_mats, sticker_centroid_base)

        # 6) flatten the 6xp blocks into one long axis
        #    shapes: (_faces:    6pxFx3), (_stickers: 6pxSx3)
        #            (_sticker_centroids: 6px3)
        self._faces: NDArray[float32] = faces_t.reshape(-1, faces_t.shape[2], 3)
        self._stickers: NDArray[float32] = stickers_t.reshape(-1, stickers_t.shape[2], 3)
        self._sticker_centroids: NDArray[float32] = stickr_ctr_t.reshape(-1, 3)

        # 7) face-centroids plus color-ID column
        fc: NDArray[float32] = face_ctr_t.reshape(-1, 3)  # (6p,3)
        colors: NDArray[np.int32] = np.arange(6 * p, dtype=np.int32)[:, None]  # (6p,1)
        # hstack promotes integer inputs to floating point.
        self._face_centroids: NDArray[float32 | np.int32] = np.hstack([fc, colors])  # (6p,4)

    def _project(self, pts: NDArray[float32]) -> NDArray[float32]:
        """Project points using the current rotation and view.

        Args:
            pts: 3D points.

        Returns:
            Projected 3D points.
        """
        return project_points(pts, self._current_rot, self._view, self.vertical_axis)

    def _compute_projections(self) -> None:
        """Compute all necessary projections for drawing.

        This is separated from _draw_cube to allow caching.
        """
        if self._proj_stickers is None:
            # Compute projections for stickers, faces, and centroids
            self._proj_stickers = self._project(self._stickers)
            self._proj_faces = self._project(self._faces)
            self._proj_face_centroids = self._project(self._face_centroids[:, :3].astype(float32))
            self._proj_sticker_centroids = self._project(self._sticker_centroids)

            # Extract 2D coordinates for drawing
            self._stickers_2d = self._proj_stickers[..., :2]
            self._faces_2d = self._proj_faces[..., :2]

            # Calculate z-orders once
            self._face_zorders = -self._proj_face_centroids[:, 2]
            self._sticker_zorders = -self._proj_sticker_centroids[:, 2]

    def _draw_cube(self) -> None:
        """Render or update the cube visualization."""
        # Compute projections if not already cached
        self._compute_projections()

        # Compute projections before reading face coordinates.
        assert self._stickers_2d is not None
        assert self._faces_2d is not None
        assert self._face_zorders is not None
        assert self._sticker_zorders is not None

        # Determine colors from face_colors and update for grey/black stickers if needed
        cube_area: int = self.N**2
        # Use asarray for better performance than a list comprehension
        colors_arr: NDArray[uint8] = np.asarray(self.face_colors)[self.colors // cube_area].copy()

        # Apply gray and black stickers. These index lists are usually short.
        for idx in self._grey_stickers:
            colors_arr[idx] = "grey"
        for idx in self._black_stickers:
            colors_arr[idx] = "k"

        if self._face_polys is None:
            # Initial call: create polygon objects and add to axes
            # Pre-allocate lists for better performance
            self._face_polys = []
            self._sticker_polys = []

            face_poly: Polygon
            sticker_poly: Polygon
            for i, color in enumerate(colors_arr):
                face_poly = Polygon(self._faces_2d[i], facecolor=self.plastic_color, zorder=self._face_zorders[i])
                sticker_poly = Polygon(self._stickers_2d[i], facecolor=color, zorder=self._sticker_zorders[i])
                self._face_polys.append(face_poly)
                self._sticker_polys.append(sticker_poly)
                self.add_patch(face_poly)
                self.add_patch(sticker_poly)
        else:
            # Subsequent call: update the polygon objects
            # This is faster than recreating polygons
            assert self._face_polys is not None
            assert self._sticker_polys is not None

            for i, color in enumerate(colors_arr):
                self._face_polys[i].set_xy(self._faces_2d[i])
                self._face_polys[i].set_zorder(self._face_zorders[i])
                self._face_polys[i].set_facecolor(self.plastic_color)

                self._sticker_polys[i].set_xy(self._stickers_2d[i])
                self._sticker_polys[i].set_zorder(self._sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(color)

        # Defer the draw until the GUI event loop is idle.
        # to only update when the GUI is idle and ready. Batch renderers set
        # defer_draws and issue an explicit canvas.draw() before reading pixels.
        if not self.defer_draws:
            self.figure.canvas.draw_idle()

    def new_state(self, colors: NDArray[uint8]) -> None:
        """Update the cube with a new color state and redraw.

        Args:
            colors: New color assignments for stickers.
        """
        # Matplotlib receives a contiguous color array.
        self.colors = np.ascontiguousarray(colors, dtype=uint8)
        self._draw_cube()
