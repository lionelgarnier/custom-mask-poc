"""
Visualization functions for 3D face models and landmarks
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyvista as pv
import vtk
from utils import create_pyvista_mesh, smooth_line_points, set_front_view
from config import DEFAULT_FACE_CONTOUR_LANDMARKS
from face import extract_face_landmarks

def visualize_2d_landmarks(image, title="Landmarks 2D"):
    """Visualize 2D landmarks on image"""
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    return fig

def visualize_3d_landmarks(points_3d, title="Landmarks 3D"):
    """Visualize 3D landmarks in a scatter plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='b', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Adjust axes for better visualization
    max_range = np.max([
        np.max(points_3d[:,0]) - np.min(points_3d[:,0]),
        np.max(points_3d[:,1]) - np.min(points_3d[:,1]),
        np.max(points_3d[:,2]) - np.min(points_3d[:,2])
    ])
    mid_x = (np.max(points_3d[:,0]) + np.min(points_3d[:,0])) * 0.5
    mid_y = (np.max(points_3d[:,1]) + np.min(points_3d[:,1])) * 0.5
    mid_z = (np.max(points_3d[:,2]) + np.min(points_3d[:,2])) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    
    return fig

def visualize_mesh_with_landmarks(plotter, mesh, landmarks, landmark_indices, keep_indices = None):
    """Visualize 3D mesh with landmarks and IDs"""

    pv_mesh = create_pyvista_mesh(mesh)
    plotter.add_mesh(pv_mesh, color='white', opacity=0.5)

    # Add landmarks with their IDs
    if keep_indices is not None:
        # Keep only landmarks corresponding to given IDs
        landmarks = np.array([landmarks[i] for i in keep_indices])
    else:
        landmarks = np.array(landmarks)

    plotter.add_points(landmarks, color='green', point_size=4, render_points_as_spheres=True)
    # for idx, point in zip(landmark_indices, landmarks):
    #     plotter.add_point_labels([point], [str(idx)], font_size=10, text_color='blue', shape_opacity=0.0)

    
    return plotter

def visualize_contact_line(mesh, line_points, plotter=None):
    """Visualize 3D mesh with contact line"""
    if plotter is None:
        plotter = pv.Plotter()
        return_plotter = True
    else:
        return_plotter = False
    
    pv_mesh = create_pyvista_mesh(mesh)
    
    # Add mesh with RGB colors if available, otherwise white
    if mesh.has_vertex_colors():
        plotter.add_mesh(pv_mesh, scalars="RGB", rgb=True, opacity=1.0)
    else:
        plotter.add_mesh(pv_mesh, color='white', opacity=1.0)

    # Add the yellow lines
    proj_line = pv.lines_from_points(np.array(line_points), close=True)
    plotter.add_mesh(proj_line, color='yellow', line_width=4)
    
    # Set to front view
    set_front_view(plotter)
    
    # Enable trackball actor style for better navigation
    plotter.enable_trackball_actor_style()
    
    if return_plotter:
        plotter.show_grid()
        return plotter
    else:
        return None


def show_face_with_landmarks_3d(face_mesh,
                                face_landmarks,
                                face_landmarks_ids=None,
                                keep_ids=None,
                                show_ids=True,
                                window_title="Face Mesh with Landmarks"):
    """
    Open an interactive PyVista window showing the 3D face mesh and landmarks.

    - Orbit/zoom enabled (trackball style)
    - Optional landmark ID labels

    Parameters:
    -----------
    face_mesh : Open3D TriangleMesh | pyvista.PolyData | dict
        The 3D face mesh (any format supported by create_pyvista_mesh)
    face_landmarks : np.ndarray (N,3)
        3D landmark coordinates (filtered valid landmarks)
    face_landmarks_ids : iterable[int] | None
        Original landmark IDs corresponding to face_landmarks rows
    keep_ids : iterable[int] | None
        Subset of original landmark IDs to display (filter)
    show_ids : bool
        Whether to display labels next to points
    window_title : str
        Window title
    """

    plotter = pv.Plotter()
    plotter.add_title(window_title)

    pv_mesh = create_pyvista_mesh(face_mesh)
    plotter.add_mesh(pv_mesh, color='white', opacity=0.5)

    # Build point set (optionally filtered by keep_ids)
    if keep_ids is not None and face_landmarks_ids is not None:
        id_to_point = {int(i): p for i, p in zip(face_landmarks_ids, face_landmarks)}
        pts = np.array([id_to_point[i] for i in keep_ids if i in id_to_point], dtype=float)
        labels = [str(i) for i in keep_ids if i in id_to_point]
    else:
        pts = np.asarray(face_landmarks, dtype=float)
        labels = [str(int(i)) for i in face_landmarks_ids] if (show_ids and face_landmarks_ids is not None) else None

    labels_actor_holder = {"actor": None}

    if pts.size:
        plotter.add_points(pts, color='red', point_size=6, render_points_as_spheres=True)
        if show_ids and labels is not None:
            try:
                actor = plotter.add_point_labels(pts, labels, font_size=12, text_color='red', shape_opacity=0.0)
                labels_actor_holder["actor"] = actor
            except Exception:
                # Fallback: per-point (slower)
                for p, lbl in zip(pts, labels):
                    plotter.add_point_labels([p], [lbl], font_size=12, text_color='red', shape_opacity=0.0)

    # Camera and interaction
    set_front_view(plotter)
    plotter.enable_trackball_style()
    # Ensure perspective projection so dolly zoom behaves as expected
    try:
        plotter.camera.SetParallelProjection(False)
    except Exception:
        pass
    # Request right-click to pan using interactor custom mapping
    try:
        if hasattr(plotter, 'iren') and plotter.iren is not None:
            plotter.iren.enable_custom_trackball_style(
                left_button='rotate',
                right_button='pan',
                middle_button='dolly',
                shift_left_button='pan',
                shift_right_button='rotate',
                shift_middle_button='dolly'
            )
    except Exception:
        pass
    # Quick zoom helpers and robust wheel bindings
    try:
        plotter.add_key_event('z', lambda: plotter.enable_zoom_style())  # Drag to zoom mode
        plotter.add_key_event('t', lambda: plotter.enable_trackball_style())  # Back to orbit mode
        plotter.add_key_event('j', lambda: plotter.enable_joystick_style())  # Joystick mode
        # Avoid '+'/'-' (often bound to point-size), use '[' and ']' instead
        plotter.add_key_event(']', lambda: (plotter.camera.zoom(1.15), plotter.render()))  # Zoom in
        plotter.add_key_event('[', lambda: (plotter.camera.zoom(0.87), plotter.render()))  # Zoom out
        # Toggle landmark IDs visibility
        def _toggle_labels():
            actor = labels_actor_holder.get("actor")
            if actor is not None:
                vis = actor.GetVisibility()
                if vis:
                    actor.VisibilityOff()
                else:
                    actor.VisibilityOn()
                plotter.render()
        plotter.add_key_event('i', _toggle_labels)
        # Mouse wheel fallback: force camera zoom on wheel events
        def _on_wheel_forward(obj=None, evt=None):
            try:
                plotter.camera.zoom(1.1)
                plotter.render()
            except Exception:
                pass
        def _on_wheel_backward(obj=None, evt=None):
            try:
                plotter.camera.zoom(0.9)
                plotter.render()
            except Exception:
                pass
        if hasattr(plotter, 'iren') and plotter.iren is not None:
            plotter.iren.add_observer('MouseWheelForwardEvent', _on_wheel_forward)
            plotter.iren.add_observer('MouseWheelBackwardEvent', _on_wheel_backward)
    except Exception:
        pass
    plotter.show_grid()
    plotter.show()

    return plotter


def preview_face_landmarks_from_mesh_path(mesh_path,
                                          keep_ids=None,
                                          show_ids=False,
                                          window_title="Face Mesh with Landmarks"):
    """
    Convenience helper: extract landmarks from a mesh path and display interactively.
    """
    mesh, landmarks, landmark_ids = extract_face_landmarks(mesh_path)
    return show_face_with_landmarks_3d(mesh,
                                       landmarks,
                                       landmark_ids,
                                       keep_ids=keep_ids,
                                       show_ids=show_ids,
                                       window_title=window_title)