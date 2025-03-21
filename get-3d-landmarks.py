import mediapipe as mp
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyvista as pv
from scipy.interpolate import splprep, splev

# Charge le modèle 3D
mesh = o3d.io.read_triangle_mesh("D:/OneDrive/Desktop/masque/face_lionel_long_capture.obj")
mesh.compute_vertex_normals()

# Configuration initiale avec une seule vue
width, height = 800, 800
vis = o3d.visualization.Visualizer()
vis.create_window(width=width, height=height, visible=False)
vis.add_geometry(mesh)
vis.get_render_option().background_color = np.asarray([0, 0, 0])
vis.poll_events()
vis.update_renderer()

# Capturer image ET profondeur depuis la même vue
depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
image_color = np.asarray(vis.capture_screen_float_buffer(do_render=True))

# Obtenir les paramètres RÉELS de la caméra virtuelle
view_control = vis.get_view_control()
camera_params = view_control.convert_to_pinhole_camera_parameters()
intrinsic = camera_params.intrinsic.intrinsic_matrix
extrinsic = camera_params.extrinsic
vis.destroy_window()

# Préparation de l'image pour MediaPipe
image_color_cv = (image_color * 255).astype(np.uint8)
image_bgr = cv2.cvtColor(image_color_cv, cv2.COLOR_RGB2BGR)
cv2.imwrite("projection.png", image_bgr)  # Sauvegarde facultative

# Détection avec MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

if not results.multi_face_landmarks:
    raise ValueError("Aucun visage détecté!")

# Récupération des landmarks et création d'une image de visualisation
landmarks = results.multi_face_landmarks[0]
viz_image = image_bgr.copy()
points_2d = []

for lm in landmarks.landmark:
    x_px = int(lm.x * width)
    y_px = int(lm.y * height)
    points_2d.append((x_px, y_px))
    cv2.circle(viz_image, (x_px, y_px), 1, (0, 255, 0), -1)

# Affichage des landmarks 2D
plt.figure()
plt.imshow(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
plt.title("Landmarks détectés par MediaPipe")
plt.axis("off")
plt.show()

# Extraction des landmarks 3D
points_3d = []
for (x_px, y_px) in points_2d:
    # Vérifier que le point est dans les limites de l'image
    if 0 <= x_px < width and 0 <= y_px < height:
        z = depth[y_px, x_px]
        
        if z > 0 and not np.isnan(z) and not np.isinf(z):
            # Utilisation des paramètres intrinsèques réels pour la déprojection
            x_normalized = (x_px - intrinsic[0, 2]) / intrinsic[0, 0]
            y_normalized = (y_px - intrinsic[1, 2]) / intrinsic[1, 1]
            
            x_3d = x_normalized * z
            y_3d = y_normalized * z
            z_3d = z
            
            # Transformation vers l'espace mondial avec la matrice extrinsèque
            point_camera = np.array([x_3d, y_3d, z_3d, 1.0])
            point_world = np.linalg.inv(extrinsic) @ point_camera
            points_3d.append(point_world[:3])
        else:
            points_3d.append([np.nan, np.nan, np.nan])
    else:
        points_3d.append([np.nan, np.nan, np.nan])

# Conversion et filtrage des points invalides
points_3d = np.array(points_3d, dtype=np.float64)
valid_mask = ~np.isnan(points_3d).any(axis=1)
valid_points_3d = points_3d[valid_mask]

# Affichage optimal des résultats
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(valid_points_3d[:,0], valid_points_3d[:,1], valid_points_3d[:,2], c='b', s=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Landmarks 3D reprojetés (approche optimale)")

# Ajuster les axes pour une meilleure visualisation
max_range = np.max([
    np.max(valid_points_3d[:,0]) - np.min(valid_points_3d[:,0]),
    np.max(valid_points_3d[:,1]) - np.min(valid_points_3d[:,1]),
    np.max(valid_points_3d[:,2]) - np.min(valid_points_3d[:,2])
])
mid_x = (np.max(valid_points_3d[:,0]) + np.min(valid_points_3d[:,0])) * 0.5
mid_y = (np.max(valid_points_3d[:,1]) + np.min(valid_points_3d[:,1])) * 0.5
mid_z = (np.max(valid_points_3d[:,2]) + np.min(valid_points_3d[:,2])) * 0.5
ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)

plt.show()

# Sauvegarde des landmarks 3D
np.save("face_landmarks_3d.npy", valid_points_3d)
print(f"Nombre de landmarks 3D valides extraits : {len(valid_points_3d)} / {len(points_2d)}")

# Visualisation combinée du modèle 3D et des landmarks
def visualize_mesh_with_landmarks(mesh, landmarks, landmark_indices=None):
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.triangles)

    # Conversion du mesh Open3D vers PyVista
    faces_pyvista = np.hstack((np.full((len(mesh_faces), 1), 3), mesh_faces)).astype(np.int64)
    pv_mesh = pv.PolyData(mesh_vertices, faces_pyvista)

    # Création du plotter PyVista avec accélération GPU native
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color='white', opacity=0.5)

    # Ajouter les landmarks avec leurs IDs
    landmarks = np.array(landmarks)
    plotter.add_points(landmarks, color='red', point_size=6, render_points_as_spheres=True)

    if landmark_indices is None:
        landmark_indices = np.arange(len(landmarks))

    for idx, point in zip(landmark_indices, landmarks):
        plotter.add_point_labels([point], [str(idx)], font_size=10, text_color='blue', shape_opacity=0.0)

    # Draw green line joining specified landmark IDs
    original_ids = [168, 417, 465, 429, 423, 391, 393, 164, 167, 165, 203, 209, 245, 193, 168]
    valid_ids = [np.where(landmark_indices == id)[0][0] for id in original_ids if id in landmark_indices]
    line_points = landmarks[valid_ids]
    line_points = smooth_line_points(line_points, smoothing=0.1, num_samples=300)
    line = pv.lines_from_points(line_points, close=True)
    plotter.add_mesh(line, color='green', line_width=2)
    
    # Project green line onto the face using a front-view projection (along Z)
    bounds = pv_mesh.bounds  # mesh bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    dz = 100  # offset for ray tracing
    projected_line_points = []
    for pt in line_points:
        origin = (pt[0], pt[1], bounds[5] + dz)
        end = (pt[0], pt[1], bounds[4] - dz)
        pts, ind = pv_mesh.ray_trace(origin, end, first_point=True)
        if pts.size:
            projected_line_points.append(pts)
        else:
            projected_line_points.append(pt)  # fallback if no intersection
    if len(projected_line_points) > 1:
        proj_line = pv.lines_from_points(np.array(projected_line_points), close=False)
        plotter.add_mesh(proj_line, color='orange', line_width=4)
    
    plotter.show_grid()
    plotter.show()

def smooth_line_points(points, smoothing=0.1, num_samples=300):
    x, y, z = points[:,0], points[:,1], points[:,2]
    tck, u = splprep([x, y, z], s=smoothing, k=2, per=True)
    u_new = np.linspace(0, 1, num_samples)
    x_new, y_new, z_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new, z_new))

def visualize_mesh_with_yellow_lines(mesh, line_points):
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.triangles)

    # Conversion du mesh Open3D vers PyVista
    faces_pyvista = np.hstack((np.full((len(mesh_faces), 1), 3), mesh_faces)).astype(np.int64)
    pv_mesh = pv.PolyData(mesh_vertices, faces_pyvista)
    
    # Transfer vertex colors if available
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        # Convert RGB colors from [0,1] to [0,255] for PyVista
        colors = (colors * 255).astype(np.uint8)
        pv_mesh.point_data["RGB"] = colors
        
    # Création du plotter PyVista avec accélération GPU native
    plotter = pv.Plotter()
    
    # Add mesh with RGB colors if available, otherwise white
    if mesh.has_vertex_colors():
        plotter.add_mesh(pv_mesh, scalars="RGB", rgb=True, opacity=1.0)
    else:
        plotter.add_mesh(pv_mesh, color='white', opacity=1.0)

    # Add the yellow lines
    proj_line = pv.lines_from_points(np.array(line_points), close=True)  # Set close=True to connect last point to first
    plotter.add_mesh(proj_line, color='yellow', line_width=4)
    
    plotter.show_grid()
    plotter.show()

# Récupérer les indices correspondant aux landmarks valides
valid_indices = np.where(valid_mask)[0]

# Generate line points before visualization (global scope)
mesh_vertices = np.asarray(mesh.vertices)
mesh_faces = np.asarray(mesh.triangles)
faces_pyvista = np.hstack((np.full((len(mesh_faces), 1), 3), mesh_faces)).astype(np.int64)
pv_mesh = pv.PolyData(mesh_vertices, faces_pyvista)

# Create line points based on landmarks
original_ids = [168, 417, 465, 429, 423, 391, 393, 164, 167, 165, 203, 209, 245, 193, 168]
valid_ids = [np.where(valid_indices == id)[0][0] for id in original_ids if id in valid_indices]
line_points = valid_points_3d[valid_ids]
line_points = smooth_line_points(line_points, smoothing=0.1, num_samples=300)

# Project the line onto the face
bounds = pv_mesh.bounds
dz = 100
projected_line_points = []
for pt in line_points:
    origin = (pt[0], pt[1], bounds[5] + dz)
    end = (pt[0], pt[1], bounds[4] - dz)
    pts, ind = pv_mesh.ray_trace(origin, end, first_point=True)
    if pts.size:
        projected_line_points.append(pts)
    else:
        projected_line_points.append(pt)

# Lancer la visualisation combinée avec les IDs
visualize_mesh_with_landmarks(mesh, valid_points_3d, valid_indices)

# Lancer la nouvelle visualisation avec les lignes jaunes
visualize_mesh_with_yellow_lines(mesh, projected_line_points)
