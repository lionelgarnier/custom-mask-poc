#!/usr/bin/env python3
"""
Test rapide du lissage conservateur pour vérifier qu'on ne casse plus la géométrie.
"""

import numpy as np
import pyvista as pv
from models.roundednose_model import RoundedNoseModel

def test_conservative_approach():
    """Test avec données minimales pour vérifier que ça marche."""
    print("🧪 TEST LISSAGE CONSERVATEUR")
    print("=" * 40)
    
    # Données très simples pour test rapide
    face_mesh = pv.Sphere(radius=50, theta_resolution=30, phi_resolution=30)
    face_mesh = face_mesh.compute_normals()
    
    # Landmarks minimaux
    face_landmarks = np.array([
        [0, 0, 35],      # ID 4: bout du nez  
        [0, 5, 40],      # ID 6: pont du nez
        [-15, -5, 25],   # ID 94: coin gauche
        [15, -5, 25],    # ID 117: coin droit
        [-15, 5, 25],    # ID 346: pommette gauche
        [15, 5, 25],     # ID 5: pommette droite
        # Contour minimal du nez
        [0, 8, 38], [-8, 3, 35], [-5, 0, 38], [-3, -3, 36], [0, -5, 35], [3, -3, 36], 
        [5, 0, 38], [8, 3, 35], [6, 6, 37], [3, 8, 38], [0, 10, 39], [-3, 8, 38]
    ])
    
    face_landmarks_ids = np.array([4, 6, 94, 117, 346, 5, 197, 437, 371, 423, 391, 393, 164, 167, 165, 203, 142, 217])
    
    # Test de différents niveaux
    model = RoundedNoseModel()
    
    levels = ["none", "light", "medium"]  # Test conservateur
    
    for level in levels:
        print(f"\n🔄 Test niveau: {level}")
        try:
            volume, message = model.create_3d_object(
                output_path="temp",
                face_mesh=face_mesh,
                face_landmarks=face_landmarks,
                face_landmarks_ids=face_landmarks_ids,
                smooth_intensity=level
            )
            
            print(f"   ✅ Succès: {volume.n_points} points, {volume.n_faces} faces")
            print(f"   📏 Volume: {volume.volume:.2f}, Manifold: {'✅' if volume.is_manifold else '❌'}")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print("\n✅ Test terminé! Le modèle devrait maintenant être plus stable.")
    print("💡 Utilisez smooth_intensity='medium' pour un lissage équilibré")
    print("💡 Utilisez smooth_intensity='none' ou 'light' pour préserver totalement la géométrie")

if __name__ == "__main__":
    test_conservative_approach() 