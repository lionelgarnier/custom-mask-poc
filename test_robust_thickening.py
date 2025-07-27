#!/usr/bin/env python3
"""
Test script for robust thickening functionality.
Demonstrates the new thicken_mesh_robust function with fallback capabilities.
"""

import numpy as np
import pyvista as pv
from utils import (thicken_mesh_robust, install_topological_offsets_guide, 
                   thicken_mesh, thicken_mesh_vtk)

def create_test_surface():
    """Create a test surface for thickening experiments."""
    # Create a sphere with some irregularities
    sphere = pv.Sphere(radius=10, theta_resolution=20, phi_resolution=20)
    
    # Add some noise to make it more challenging
    points = sphere.points
    noise = np.random.normal(0, 0.5, points.shape)
    sphere.points = points + noise
    
    # Compute normals
    sphere = sphere.compute_normals()
    
    return sphere

def test_thickening_methods():
    """Test different thickening methods and compare results."""
    print("🧪 Test des méthodes d'épaississement\n")
    
    # Create test surface
    surface = create_test_surface()
    thickness = 2.0
    
    results = {}
    timings = {}
    
    # Test 1: Topological Offsets (will likely fail without installation)
    print("1️⃣ Test Topological Offsets...")
    try:
        import time
        start = time.time()
        result_topo = thicken_mesh_robust(surface, thickness, method="topological_offsets", fallback=False)
        timings['topological_offsets'] = time.time() - start
        results['topological_offsets'] = result_topo
        print(f"   ✅ Succès en {timings['topological_offsets']:.2f}s")
        print(f"   📊 {result_topo.n_points} points, {result_topo.n_faces} faces")
        print(f"   🔍 Manifold: {result_topo.is_manifold}")
    except Exception as e:
        print(f"   ❌ Échec: {e}")
        print("   💡 Utilisez install_topological_offsets_guide() pour l'installation")
    
    # Test 2: MRMeshPy (current method)
    print("\n2️⃣ Test MRMeshPy...")
    try:
        start = time.time()
        result_mr = thicken_mesh_robust(surface, thickness, method="mrmeshpy", fallback=False)
        timings['mrmeshpy'] = time.time() - start
        results['mrmeshpy'] = result_mr
        print(f"   ✅ Succès en {timings['mrmeshpy']:.2f}s")
        print(f"   📊 {result_mr.n_points} points, {result_mr.n_faces} faces")
        print(f"   🔍 Manifold: {result_mr.is_manifold}")
    except Exception as e:
        print(f"   ❌ Échec: {e}")
    
    # Test 3: VTK Extrude (fallback method)
    print("\n3️⃣ Test VTK Extrude...")
    try:
        start = time.time()
        result_vtk = thicken_mesh_robust(surface, thickness, method="vtk_extrude", fallback=False)
        timings['vtk_extrude'] = time.time() - start
        results['vtk_extrude'] = result_vtk
        print(f"   ✅ Succès en {timings['vtk_extrude']:.2f}s")
        print(f"   📊 {result_vtk.n_points} points, {result_vtk.n_cells} cells")
        print(f"   🔍 Type: {type(result_vtk).__name__}")
    except Exception as e:
        print(f"   ❌ Échec: {e}")
    
    # Test 4: Robust method with fallback
    print("\n4️⃣ Test méthode robuste avec fallback...")
    try:
        start = time.time()
        result_robust = thicken_mesh_robust(surface, thickness, method="topological_offsets", fallback=True)
        timings['robust'] = time.time() - start
        results['robust'] = result_robust
        print(f"   ✅ Succès en {timings['robust']:.2f}s")
        print(f"   📊 {result_robust.n_points} points, {result_robust.n_faces} faces")
        print(f"   🔍 Manifold: {result_robust.is_manifold}")
    except Exception as e:
        print(f"   ❌ Échec: {e}")
    
    return surface, results, timings

def visualize_results(surface, results):
    """Visualize comparison of different thickening methods."""
    if not results:
        print("❌ Aucun résultat à visualiser")
        return
    
    print(f"\n🎨 Visualisation de {len(results)} méthodes...")
    
    # Create subplots
    n_methods = len(results)
    plotter = pv.Plotter(shape=(1, n_methods + 1), window_size=(300 * (n_methods + 1), 400))
    
    # Original surface
    plotter.subplot(0, 0)
    plotter.add_mesh(surface, color='lightblue', show_edges=True, opacity=0.8)
    plotter.add_title("Surface originale")
    
    # Results
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, (method, result) in enumerate(results.items()):
        plotter.subplot(0, i + 1)
        plotter.add_mesh(result, color=colors[i % len(colors)], show_edges=True, opacity=0.8)
        # Handle different mesh types
        if hasattr(result, 'n_faces'):
            plotter.add_title(f"{method}\n{result.n_points} pts, {result.n_faces} faces")
        else:
            plotter.add_title(f"{method}\n{result.n_points} pts, {result.n_cells} cells")
    
    plotter.link_views()
    plotter.show()

def compare_mesh_quality(results):
    """Compare mesh quality metrics."""
    if not results:
        print("❌ Aucun résultat à comparer")
        return
    
    print("\n📈 COMPARAISON QUALITÉ DES MAILLAGES")
    print("=" * 50)
    
    for method, mesh in results.items():
        print(f"\n🔍 {method.upper()}:")
        print(f"   Points: {mesh.n_points}")
        
        # Handle different mesh types
        if hasattr(mesh, 'n_faces'):
            print(f"   Faces: {mesh.n_faces}")
        else:
            print(f"   Cells: {mesh.n_cells}")
        
        # Only compute volume/area/manifold for PolyData
        if hasattr(mesh, 'volume'):
            print(f"   Volume: {mesh.volume:.2f}")
        if hasattr(mesh, 'area'):
            print(f"   Surface area: {mesh.area:.2f}")
        if hasattr(mesh, 'is_manifold'):
            print(f"   Manifold: {'✅' if mesh.is_manifold else '❌'}")
        else:
            print(f"   Type: {type(mesh).__name__}")
        
        # Check for degenerate triangles
        try:
            quality = mesh.compute_cell_quality()
            min_quality = quality.min()
            avg_quality = quality.mean()
            print(f"   Qualité min: {min_quality:.4f}")
            print(f"   Qualité moy: {avg_quality:.4f}")
        except:
            print("   Qualité: N/A")

def main():
    """Main test function."""
    print("🚀 TEST DE L'ÉPAISSISSEMENT ROBUSTE")
    print("=" * 40)
    
    # Run tests
    surface, results, timings = test_thickening_methods()
    
    # Compare quality
    compare_mesh_quality(results)
    
    # Show installation guide if Topological Offsets failed
    if 'topological_offsets' not in results:
        print("\n" + "=" * 50)
        install_topological_offsets_guide()
    
    # Visualize if we have results
    visualize_results(surface, results)
    
    print("\n✅ Tests terminés!")

if __name__ == "__main__":
    main() 