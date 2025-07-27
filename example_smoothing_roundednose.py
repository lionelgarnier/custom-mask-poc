#!/usr/bin/env python3
"""
Exemple d'utilisation des nouvelles options de lissage pour le modèle RoundedNose.
Montre comment réduire la visibilité des facettes avec différents niveaux d'intensité.
"""

import numpy as np
import pyvista as pv
from models.roundednose_model import RoundedNoseModel
import os

def create_example_face_data():
    """Créer des données d'exemple pour tester le lissage."""
    # Simuler un visage simple
    face_mesh = pv.Sphere(radius=50, center=(0, 0, 0), theta_resolution=50, phi_resolution=50)
    
    # Aplatir légèrement pour ressembler à un visage
    points = face_mesh.points
    points[:, 2] *= 0.8  # Aplatir dans la direction Z
    face_mesh.points = points
    face_mesh = face_mesh.compute_normals()
    
    # Créer des landmarks d'exemple (positions approximatives sur un visage)
    face_landmarks = np.array([
        [0, 0, 35],      # ID 4: bout du nez
        [0, 5, 40],      # ID 6: pont du nez
        [-15, -5, 25],   # ID 94: coin gauche
        [15, -5, 25],    # ID 117: coin droit  
        [-15, 5, 25],    # ID 346: pommette gauche
        [15, 5, 25],     # ID 5: pommette droite
        # Contour du nez
        [0, 8, 38],      # ID 197
        [-8, 3, 35],     # ID 437
        [-5, 0, 38],     # ID 371
        [-3, -3, 36],    # ID 423
        [0, -5, 35],     # ID 391
        [3, -3, 36],     # ID 393
        [5, 0, 38],      # ID 164
        [8, 3, 35],      # ID 167
        [6, 6, 37],      # ID 165
        [3, 8, 38],      # ID 203
        [0, 10, 39],     # ID 142
        [-3, 8, 38],     # ID 217
    ])
    
    face_landmarks_ids = np.array([4, 6, 94, 117, 346, 5, 197, 437, 371, 423, 391, 393, 164, 167, 165, 203, 142, 217])
    
    return face_mesh, face_landmarks, face_landmarks_ids

def demonstrate_smoothing_levels():
    """Démonstrer les différents niveaux de lissage."""
    print("🎨 DÉMONSTRATION DES NIVEAUX DE LISSAGE ROUNDEDNOSE")
    print("=" * 60)
    
    # Créer les données d'exemple
    face_mesh, face_landmarks, face_landmarks_ids = create_example_face_data()
    
    # Créer le modèle
    rounded_nose_model = RoundedNoseModel()
    
    # Tester différents niveaux de lissage
    smoothing_levels = ["light", "medium", "strong", "very_strong"]
    results = {}
    
    for level in smoothing_levels:
        print(f"\n🔄 Test du niveau de lissage: {level.upper()}")
        
        try:
            import time
            start = time.time()
            
            volume, message = rounded_nose_model.create_3d_object(
                output_path="temp",
                face_mesh=face_mesh,
                face_landmarks=face_landmarks,
                face_landmarks_ids=face_landmarks_ids,
                smooth_intensity=level
            )
            
            processing_time = time.time() - start
            
            print(f"   ✅ Succès en {processing_time:.2f}s")
            print(f"   📊 Résultat: {volume.n_points} points, {volume.n_faces} faces")
            print(f"   🔍 Manifold: {'✅' if volume.is_manifold else '❌'}")
            
            results[level] = {
                'volume': volume,
                'time': processing_time,
                'n_points': volume.n_points,
                'n_faces': volume.n_faces,
                'is_manifold': volume.is_manifold
            }
            
        except Exception as e:
            print(f"   ❌ Échec: {e}")
            results[level] = None
    
    return results, face_mesh

def visualize_smoothing_comparison(results, face_mesh):
    """Visualiser la comparaison des différents niveaux de lissage."""
    print("\n🎨 Visualisation comparative des niveaux de lissage...")
    
    # Filtrer les résultats réussis
    successful_results = {k: v for k, v in results.items() if v is not None}
    n_results = len(successful_results)
    
    if n_results == 0:
        print("❌ Aucun résultat à visualiser")
        return
    
    # Créer la grille de visualisation
    cols = min(3, n_results)
    rows = (n_results + cols - 1) // cols
    
    plotter = pv.Plotter(shape=(rows, cols), window_size=(400 * cols, 400 * rows))
    
    # Couleurs pour chaque niveau
    colors = {
        'light': 'lightblue',
        'medium': 'lightgreen', 
        'strong': 'orange',
        'very_strong': 'red'
    }
    
    row, col = 0, 0
    for level, result in successful_results.items():
        plotter.subplot(row, col)
        
        volume = result['volume']
        color = colors.get(level, 'gray')
        
        # Afficher le volume avec edges pour voir les facettes
        plotter.add_mesh(volume, color=color, show_edges=True, edge_color='black', 
                        line_width=0.5, opacity=0.8)
        plotter.add_mesh(face_mesh, color='lightgray', opacity=0.2)
        
        manifold_icon = "✅" if result['is_manifold'] else "❌"
        title = f"{level.upper()}\n{result['n_faces']} faces\nManifold: {manifold_icon}"
        plotter.add_title(title)
        
        # Passer à la cellule suivante
        col += 1
        if col >= cols:
            col = 0
            row += 1
    
    plotter.link_views()
    plotter.show()

def save_smoothing_comparison(results, output_dir="output/smoothing_comparison"):
    """Sauvegarder les résultats pour comparaison ultérieure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n💾 Sauvegarde des résultats dans: {output_dir}")
    
    for level, result in results.items():
        if result is not None:
            output_path = os.path.join(output_dir, f"roundednose_{level}_smoothing.stl")
            result['volume'].save(output_path)
            print(f"   ✅ {level}: {output_path}")

def print_smoothing_guide():
    """Afficher un guide d'utilisation des niveaux de lissage."""
    guide = """
📚 GUIDE DES NIVEAUX DE LISSAGE ROUNDEDNOSE (APPROCHE CONSERVATRICE)
====================================================================

🔹 NONE (Aucun lissage):
   • Aucun lissage supplémentaire, juste repair de base
   • Recommandé pour: préserver exactement la géométrie originale
   • Facettes: Visibles mais géométrie parfaitement préservée

🔹 LIGHT (Léger) - PAR DÉFAUT:
   • Lissage minimal, préserve tous les détails et épaisseurs
   • Recommandé pour: usage général, prototypes
   • Facettes: Légèrement réduites, géométrie préservée

🔸 MEDIUM (Moyen):
   • Lissage équilibré sans affecter la géométrie principale
   • Recommandé pour: réduire les facettes visibles en sécurité
   • Facettes: Bien réduites, épaisseurs préservées

🔶 STRONG (Fort):
   • Lissage plus important mais encore contrôlé
   • Recommandé pour: rendu visuel avec prudence
   • Facettes: Fortement réduites

🔴 VERY_STRONG (Très fort):
   • Lissage maximum mais bridé pour éviter la dégradation
   • Recommandé pour: cas spécifiques seulement
   • ⚠️  À utiliser avec précaution

💻 UTILISATION RECOMMANDÉE:
===========================

# Pour préserver la géométrie (par défaut):
volume, message = rounded_nose_model.create_3d_object(
    output_path="output/my_nose.stl",
    face_mesh=face_mesh,
    face_landmarks=face_landmarks,
    face_landmarks_ids=face_landmarks_ids,
    smooth_intensity="light"  # ← Valeur par défaut sûre
)

# Pour réduire les facettes en sécurité:
volume, message = rounded_nose_model.create_3d_object(
    output_path="output/my_nose.stl",
    face_mesh=face_mesh,
    face_landmarks=face_landmarks,
    face_landmarks_ids=face_landmarks_ids,
    smooth_intensity="medium"  # ← Équilibré et sûr
)

🎯 RECOMMANDATIONS CONSERVATRICES:
==================================
• Pour préserver la géométrie: "none" ou "light"
• Pour un équilibre sûr: "medium"
• Pour l'impression 3D: "medium" (testé)
• Pour l'analyse: "light" ou "none"
• ⚠️  Éviter "strong" et "very_strong" sauf test préalable

🔧 EN CAS DE PROBLÈME:
======================
Si le lissage cause des problèmes d'épaisseur ou de géométrie:
1. Utilisez smooth_intensity="none" 
2. Ou smooth_intensity="light" (par défaut)
3. Le modèle restera fonctionnel avec facettes visibles mais géométrie correcte
"""
    print(guide)

def main():
    """Fonction principale."""
    print("🚀 DÉMONSTRATION LISSAGE ROUNDEDNOSE")
    print("=" * 45)
    
    # Démontrer les niveaux de lissage
    results, face_mesh = demonstrate_smoothing_levels()
    
    # Visualiser la comparaison
    visualize_smoothing_comparison(results, face_mesh)
    
    # Sauvegarder les résultats
    save_smoothing_comparison(results)
    
    # Afficher le guide
    print_smoothing_guide()
    
    print("\n✅ Démonstration terminée!")
    print("\n💡 PROCHAINES ÉTAPES:")
    print("   1. Choisir le niveau de lissage adapté à vos besoins")
    print("   2. Utiliser smooth_intensity='strong' pour réduire les facettes")
    print("   3. Tester sur vos propres données de visage")

if __name__ == "__main__":
    main() 