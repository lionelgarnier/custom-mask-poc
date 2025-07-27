#!/usr/bin/env python3
"""
Exemple de migration du modèle roundednose vers l'épaississement robuste.
Montre comment remplacer thicken_mesh_vtk par thicken_mesh_robust.
"""

import numpy as np
import pyvista as pv
from utils import thicken_mesh_robust, thicken_mesh_vtk, install_topological_offsets_guide

def roundednose_migration_example():
    """
    Exemple montrant la migration de roundednose vers l'épaississement robuste.
    """
    print("🔄 MIGRATION ROUNDEDNOSE VERS ÉPAISSISSEMENT ROBUSTE")
    print("=" * 60)
    
    # Simuler une surface du modèle roundednose (remplacez par votre surface réelle)
    surface = create_example_nose_surface()
    thickness = 1.3  # Valeur utilisée dans roundednose_model.py ligne 150
    
    print(f"📏 Surface originale: {surface.n_points} points, {surface.n_faces} faces")
    print(f"📐 Épaisseur: {thickness} mm")
    
    # ANCIENNE MÉTHODE (ligne 150 de roundednose_model.py)
    print("\n1️⃣ Ancienne méthode (thicken_mesh_vtk):")
    try:
        import time
        start = time.time()
        volume_old = thicken_mesh_vtk(surface, thickness)
        time_old = time.time() - start
        
        print(f"   ✅ Succès en {time_old:.2f}s")
        print(f"   📊 Résultat: {volume_old.n_points} points, {volume_old.n_faces} faces")
        print(f"   🔍 Manifold: {'✅' if volume_old.is_manifold else '❌'}")
        print(f"   📏 Volume: {volume_old.volume:.2f}")
        
    except Exception as e:
        print(f"   ❌ Échec: {e}")
        volume_old = None
        time_old = None
    
    # NOUVELLE MÉTHODE ROBUSTE
    print("\n2️⃣ Nouvelle méthode robuste:")
    try:
        start = time.time()
        volume_new = thicken_mesh_robust(
            surface, 
            thickness, 
            method="topological_offsets",  # Méthode la plus robuste
            fallback=True  # Fallback vers MRMeshPy puis VTK si nécessaire
        )
        time_new = time.time() - start
        
        print(f"   ✅ Succès en {time_new:.2f}s")
        print(f"   📊 Résultat: {volume_new.n_points} points, {volume_new.n_faces} faces")
        print(f"   🔍 Manifold: {'✅' if volume_new.is_manifold else '❌'}")
        print(f"   📏 Volume: {volume_new.volume:.2f}")
        
        # Comparaison si les deux ont réussi
        if volume_old is not None:
            print(f"\n📈 COMPARAISON:")
            print(f"   Gain en qualité manifold: {'✅' if volume_new.is_manifold and not volume_old.is_manifold else '➖'}")
            print(f"   Différence de volume: {abs(volume_new.volume - volume_old.volume):.2f}")
            print(f"   Différence de temps: {time_new - time_old:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Échec: {e}")
        volume_new = None
    
    # Visualisation comparative
    if volume_old is not None or volume_new is not None:
        visualize_comparison(surface, volume_old, volume_new)
    
    return volume_old, volume_new

def create_example_nose_surface():
    """Créer une surface d'exemple similaire à celle du nez dans roundednose."""
    # Créer une forme ellipsoïdale pour simuler un nez
    sphere = pv.Sphere(radius=8, theta_resolution=30, phi_resolution=30)
    
    # Déformer pour ressembler à un nez
    points = sphere.points
    
    # Allonger sur l'axe Z (nez qui dépasse)
    points[:, 2] *= 1.5
    
    # Rétrécir sur l'axe Y (nez plus fin)  
    points[:, 1] *= 0.7
    
    # Ajouter une courbure
    points[:, 2] += 0.1 * points[:, 0]**2
    
    # Créer quelques irrégularités (narines, etc.)
    for i, point in enumerate(points):
        if point[2] > 0 and abs(point[1]) < 2:  # Zone des narines
            noise = np.random.normal(0, 0.3)
            points[i, 2] += noise
    
    sphere.points = points
    sphere = sphere.compute_normals()
    
    return sphere

def visualize_comparison(surface, volume_old, volume_new):
    """Visualiser la comparaison entre ancienne et nouvelle méthode."""
    print("\n🎨 Visualisation comparative...")
    
    n_plots = 1 + (volume_old is not None) + (volume_new is not None)
    plotter = pv.Plotter(shape=(1, n_plots), window_size=(400 * n_plots, 400))
    
    # Surface originale
    plot_idx = 0
    plotter.subplot(0, plot_idx)
    plotter.add_mesh(surface, color='lightblue', show_edges=True, opacity=0.8)
    plotter.add_title("Surface originale")
    plot_idx += 1
    
    # Ancienne méthode
    if volume_old is not None:
        plotter.subplot(0, plot_idx)
        color = 'lightcoral' if not volume_old.is_manifold else 'lightgreen'
        plotter.add_mesh(volume_old, color=color, show_edges=True, opacity=0.7)
        manifold_status = "✅" if volume_old.is_manifold else "❌"
        plotter.add_title(f"Ancienne (VTK)\nManifold: {manifold_status}")
        plot_idx += 1
    
    # Nouvelle méthode
    if volume_new is not None:
        plotter.subplot(0, plot_idx)
        color = 'lightcoral' if not volume_new.is_manifold else 'darkgreen'
        plotter.add_mesh(volume_new, color=color, show_edges=True, opacity=0.7)
        manifold_status = "✅" if volume_new.is_manifold else "❌"
        plotter.add_title(f"Nouvelle (Robuste)\nManifold: {manifold_status}")
    
    plotter.link_views()
    plotter.show()

def show_migration_code():
    """Afficher le code de migration à appliquer."""
    print("\n💻 CODE DE MIGRATION POUR ROUNDEDNOSE_MODEL.PY")
    print("=" * 50)
    
    old_code = '''# ANCIEN CODE (ligne 150):
volume = thicken_mesh_vtk(surface, 1.3)'''
    
    new_code = '''# NOUVEAU CODE (plus robuste):
from utils import thicken_mesh_robust

# Option 1: Topological Offsets avec fallback
volume = thicken_mesh_robust(surface, 1.3, method="topological_offsets", fallback=True)

# Option 2: Forcer MRMeshPy (si Topological Offsets non disponible)
# volume = thicken_mesh_robust(surface, 1.3, method="mrmeshpy", fallback=True)

# Option 3: Garder VTK comme avant
# volume = thicken_mesh_robust(surface, 1.3, method="vtk_extrude", fallback=False)'''
    
    print(old_code)
    print("\n" + "🔄" * 20)
    print(new_code)
    
    benefits = """
🎯 AVANTAGES DE LA MIGRATION:

✅ Garantie manifold avec Topological Offsets
✅ Fallback automatique si une méthode échoue  
✅ Meilleure gestion des géométries complexes
✅ Preservation des sharp features
✅ Compatible avec le code existant
✅ Même interface, résultats plus robustes
"""
    print(benefits)

def main():
    """Fonction principale."""
    print("🚀 DÉMONSTRATION MIGRATION ROUNDEDNOSE")
    print("=" * 45)
    
    # Tester la migration
    volume_old, volume_new = roundednose_migration_example()
    
    # Afficher le code de migration
    show_migration_code()
    
    # Guide d'installation si nécessaire
    if volume_new is None:
        print("\n" + "=" * 50)
        install_topological_offsets_guide()
    
    print("\n✅ Démonstration terminée!")
    print("\n💡 PROCHAINES ÉTAPES:")
    print("   1. Installer Topological Offsets (optionnel)")
    print("   2. Remplacer thicken_mesh_vtk par thicken_mesh_robust")
    print("   3. Tester sur vos géométries complexes")
    print("   4. Bénéficier de maillages manifold garantis!")

if __name__ == "__main__":
    main() 