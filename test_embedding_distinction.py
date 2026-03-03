"""
Script de verificación: compara embeddings CLIP-only vs. Híbrido
para las imágenes de example_images/ y muestra la mejora en distinción.

Uso:
    cd d:\\Aplicaciones\\busqueda-de-imagenes\\backend
    python -m test_embedding_distinction
"""

import os
import sys
import numpy as np
from PIL import Image

# Agregar el directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.model import ImageEmbedder, TOTAL_DIM, CLIP_DIM

IMAGE_FOLDER = "./example_images"


def load_images(folder: str):
    """Carga todas las imágenes de la carpeta."""
    images = {}
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                images[filename] = img
            except Exception as e:
                print(f"  ⚠ Error cargando {filename}: {e}")
    return images


def l2_distance_matrix(embeddings: dict) -> dict:
    """Calcula la matriz de distancias L2 entre todos los pares."""
    names = list(embeddings.keys())
    matrix = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                dist = float(np.linalg.norm(embeddings[n1] - embeddings[n2]))
                matrix[(n1, n2)] = dist
    return matrix


def print_stats(matrix: dict, label: str):
    """Imprime estadísticas de una matriz de distancias."""
    if not matrix:
        print(f"  No hay pares para {label}")
        return

    dists = list(matrix.values())
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Pares totales:    {len(dists)}")
    print(f"  Distancia mínima: {min(dists):.6f}")
    print(f"  Distancia máxima: {max(dists):.6f}")
    print(f"  Distancia media:  {np.mean(dists):.6f}")
    print(f"  Desv. estándar:   {np.std(dists):.6f}")

    # Top 5 pares más cercanos
    sorted_pairs = sorted(matrix.items(), key=lambda x: x[1])
    print(f"\n  Top 5 pares más cercanos:")
    for (n1, n2), dist in sorted_pairs[:5]:
        short1 = n1[-20:]
        short2 = n2[-20:]
        print(f"    {short1} ↔ {short2}: {dist:.6f}")


def main():
    print("\n" + "="*60)
    print("  VERIFICACIÓN DE EMBEDDING HÍBRIDO")
    print("="*60)

    # Cargar imágenes
    print(f"\nCargando imágenes de: {IMAGE_FOLDER}")
    images = load_images(IMAGE_FOLDER)
    print(f"  Imágenes cargadas: {len(images)}")

    if len(images) < 2:
        print("  ⚠ Se necesitan al menos 2 imágenes para comparar.")
        return

    # Inicializar embedder
    print("\nInicializando modelo CLIP...")
    embedder = ImageEmbedder()

    # --- Generar embeddings CLIP-only ---
    print("\n--- Generando embeddings CLIP-only (512 dims) ---")
    clip_embeddings = {}
    for name, img in images.items():
        emb = embedder.get_clip_embedding(img)
        clip_embeddings[name] = emb
        print(f"  ✓ {name}: {emb.shape}")

    # --- Generar embeddings híbridos ---
    print(f"\n--- Generando embeddings HÍBRIDOS ({TOTAL_DIM} dims) ---")
    hybrid_embeddings = {}
    for name, img in images.items():
        emb = embedder.get_combined_embedding(img)
        hybrid_embeddings[name] = emb
        print(f"  ✓ {name}: {emb.shape}")

    # --- Calcular matrices de distancia ---
    clip_matrix = l2_distance_matrix(clip_embeddings)
    hybrid_matrix = l2_distance_matrix(hybrid_embeddings)

    print_stats(clip_matrix, "CLIP-only (512 dims)")
    print_stats(hybrid_matrix, f"HÍBRIDO ({TOTAL_DIM} dims)")

    # --- Comparar mejora ---
    clip_dists = list(clip_matrix.values())
    hybrid_dists = list(hybrid_matrix.values())

    if clip_dists and hybrid_dists:
        clip_min = min(clip_dists)
        hybrid_min = min(hybrid_dists)
        improvement = ((hybrid_min - clip_min) / max(clip_min, 1e-10)) * 100

        print(f"\n{'='*60}")
        print(f"  RESUMEN DE MEJORA")
        print(f"{'='*60}")
        print(f"  Dist. mínima CLIP-only:  {clip_min:.6f}")
        print(f"  Dist. mínima HÍBRIDO:    {hybrid_min:.6f}")
        print(f"  Mejora en separación:    {improvement:+.1f}%")

        if hybrid_min > clip_min:
            print(f"\n  ✅ El embedding híbrido MEJORA la distinción entre imágenes similares.")
        else:
            print(f"\n  ⚠ El embedding híbrido no mejoró la separación mínima.")
            print(f"    Considere ajustar los pesos o el tamaño de la grilla.")

    print()


if __name__ == "__main__":
    main()
