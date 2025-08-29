#!/usr/bin/env python3
"""
API REST para el sistema RAG de PC Parts
Proporciona endpoints para búsqueda, estadísticas y chat
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import render_template

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity

# Configuración
app = Flask(__name__)
CORS(app)  # Permitir CORS para frontend

class PCPartsRAGAPI:
    def __init__(self, base_url="http://localhost:11434", model="gemma:2b"):
        self.base_url = base_url
        self.model = model
        self.df = None
        self.embeddings_matrix = None
        self.faiss_index = None
        self.product_texts = []
        self.embeddings_cache = {}
        self.is_loaded = False
        
    def load_system(self, csv_file="partes_pc.csv", vectors_file="pc_parts_vectors"):
        """Carga el sistema completo al iniciar la API"""
        try:
            # Cargar CSV desde la raíz
            self.df = pd.read_csv(csv_file)
            print(f"✅ CSV cargado: {len(self.df)} productos")
            
            # Cargar vectores desde la carpeta ./vectores
            vectors_path = os.path.join("vectores", vectors_file)
            if self._load_vectors(vectors_path):
                self.is_loaded = True
                print("✅ API lista para recibir consultas")
                return True
            else:
                print("❌ No se pudieron cargar vectores")
                return False
                
        except Exception as e:
            print(f"❌ Error inicializando API: {e}")
            return False
    
    def _load_vectors(self, base_filename):
        """Carga vectores desde archivos"""
        # Intentar FAISS primero
        if FAISS_AVAILABLE:
            faiss_file = f"{base_filename}.faiss"
            metadata_file = f"{base_filename}_metadata.pkl"
            
            if os.path.exists(faiss_file) and os.path.exists(metadata_file):
                try:
                    self.faiss_index = faiss.read_index(faiss_file)
                    with open(metadata_file, 'rb') as f:
                        data = pickle.load(f)
                    self.product_texts = data["product_texts"]
                    print(f"🚀 FAISS cargado: {self.faiss_index.ntotal} vectores")
                    return True
                except Exception as e:
                    print(f"⚠️ Error FAISS: {e}")
        
        # Fallback a pickle
        pickle_file = f"{base_filename}.pkl"
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                self.embeddings_matrix = np.array(data["embeddings_matrix"])
                self.embeddings_cache = data.get("embeddings_cache", {})
                self.product_texts = data["product_texts"]
                print(f"💾 Pickle cargado: {len(self.embeddings_matrix)} vectores")
                return True
            except Exception as e:
                print(f"❌ Error pickle: {e}")
        
        return False   
    def get_embedding(self, text):
        """Obtiene embedding de Ollama con cache"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
            
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings", 
                json={"model": self.model, "prompt": text}, 
                timeout=500
            )
            if response.status_code == 200:
                embedding = response.json()["embedding"]
                self.embeddings_cache[text] = embedding
                return embedding
            else:
                return None
        except Exception as e:
            print(f"❌ Error embedding: {e}")
            return None
    
    def search_products(self, query, top_k=5):
        """Busca productos similares"""
        if not self.is_loaded or not query.strip():
            return []
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Usar FAISS si disponible
        if FAISS_AVAILABLE and self.faiss_index is not None:
            return self._search_faiss(query_embedding, top_k)
        # Fallback sklearn
        elif self.embeddings_matrix is not None:
            return self._search_sklearn(query_embedding, top_k)
        else:
            return []
    
    def _search_faiss(self, query_embedding, top_k):
        """Búsqueda con FAISS"""
        query_vec = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_vec)
        
        similarities, indices = self.faiss_index.search(query_vec, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            similarity = float(similarities[0][i])
            
            if idx < len(self.df):
                row = self.df.iloc[idx]
                results.append({
                    "codigo": int(row['COD']),
                    "categoria": row['CATEGORIA'],
                    "producto": row['PRODUCTO'],
                    "precio": int(row['PRECIO_MILES']),
                    "precio_formatted": f"${row['PRECIO_MILES']}k",
                    "similarity": round(similarity, 4),
                    "index": int(idx)
                })
        
        return results
    
    def _search_sklearn(self, query_embedding, top_k):
        """Búsqueda con sklearn"""
        query_vec = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vec, self.embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.df):
                row = self.df.iloc[idx]
                results.append({
                    "codigo": int(row['COD']),
                    "categoria": row['CATEGORIA'],
                    "producto": row['PRODUCTO'],
                    "precio": int(row['PRECIO_MILES']),
                    "precio_formatted": f"${row['PRECIO_MILES']}k",
                    "similarity": round(float(similarities[idx]), 4),
                    "index": int(idx)
                })
        
        return results
    
    def chat_with_context(self, query, top_k=3):
        """Chat con contexto de productos"""
        products = self.search_products(query, top_k)
        
        if not products:
            return {
                "response": "No se encontraron productos relevantes para tu consulta.",
                "products_used": [],
                "error": False
            }
        
        # Crear contexto
        context_parts = []
        for prod in products:
            context_parts.append(
                f"{prod['producto']} (Código: {prod['codigo']}, "
                f"Categoría: {prod['categoria']}, Precio: {prod['precio_formatted']})"
            )
        context = "\n".join(context_parts)
        
        # Prompt optimizado
        prompt = f"""Productos relevantes para "{query}":
{context}

Responde brevemente sobre estos productos mencionando códigos y precios específicos."""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 200,
                        "temperature": 0.7
                    }
                },
                timeout=500
            )
            
            if response.status_code == 200:
                chat_response = response.json()["response"]
                return {
                    "response": chat_response,
                    "products_used": products,
                    "error": False
                }
            else:
                return {
                    "response": "Error al generar respuesta del chat.",
                    "products_used": products,
                    "error": True
                }
                
        except requests.exceptions.Timeout:
            return {
                "response": "El chat tardó demasiado en responder, pero aquí tienes los productos encontrados.",
                "products_used": products,
                "error": True
            }
        except Exception as e:
            return {
                "response": f"Error en el chat: {str(e)}",
                "products_used": products,
                "error": True
            }
    
    def get_stats(self):
        """Obtiene estadísticas del catálogo"""
        if not self.is_loaded:
            return {}
        
        stats = {
            "total_products": len(self.df),
            "total_categories": len(self.df['CATEGORIA'].unique()),
            "price_stats": {
                "min": int(self.df['PRECIO_MILES'].min()),
                "max": int(self.df['PRECIO_MILES'].max()),
                "average": round(self.df['PRECIO_MILES'].mean(), 2)
            },
            "categories": [],
            "price_ranges": {}
        }
        
        # Estadísticas por categoría
        cat_stats = self.df['CATEGORIA'].value_counts().head(10)
        for cat, count in cat_stats.items():
            cat_df = self.df[self.df['CATEGORIA'] == cat]
            stats["categories"].append({
                "name": cat,
                "count": int(count),
                "avg_price": round(cat_df['PRECIO_MILES'].mean(), 2)
            })
        
        # Rangos de precio
        ranges = [
            (0, 100, "Muy barato"),
            (100, 300, "Barato"),
            (300, 1000, "Medio"),
            (1000, 3000, "Caro"),
            (3000, float('inf'), "Muy caro")
        ]
        
        for min_price, max_price, label in ranges:
            if max_price == float('inf'):
                count = len(self.df[self.df['PRECIO_MILES'] >= min_price])
            else:
                count = len(self.df[(self.df['PRECIO_MILES'] >= min_price) & 
                                 (self.df['PRECIO_MILES'] < max_price)])
            stats["price_ranges"][label] = count
        
        return stats
    
    def get_products_by_category(self, category, limit=20):
        """Obtiene productos por categoría"""
        if not self.is_loaded:
            return []
        
        filtered_df = self.df[self.df['CATEGORIA'].str.contains(category, case=False, na=False)]
        
        products = []
        for _, row in filtered_df.head(limit).iterrows():
            products.append({
                "codigo": int(row['COD']),
                "categoria": row['CATEGORIA'],
                "producto": row['PRODUCTO'],
                "precio": int(row['PRECIO_MILES']),
                "precio_formatted": f"${row['PRECIO_MILES']}k"
            })
        
        return products

# Instancia global de la API
rag_api = PCPartsRAGAPI()

# ENDPOINTS DE LA API

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de la API"""
    return jsonify({
        "status": "ok" if rag_api.is_loaded else "loading",
        "message": "API funcionando correctamente" if rag_api.is_loaded else "API cargando datos...",
        "timestamp": datetime.now().isoformat(),
        "ollama_url": rag_api.base_url,
        "model": rag_api.model
    })

@app.route('/search', methods=['GET', 'POST'])
def search_products():
    """Endpoint principal de búsqueda"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data or 'query' not in data:
                return jsonify({"error": "Se requiere el campo 'query' en el JSON"}), 400
            query = data['query'].strip()
            top_k = data.get('top_k', 5)
        else:  # GET
            query = request.args.get('query', '').strip()
            top_k = request.args.get('top_k', default=5, type=int)

        if not query:
            return jsonify({"error": "La consulta no puede estar vacía"}), 400
        if not rag_api.is_loaded:
            return jsonify({"error": "API no inicializada correctamente"}), 503

        results = rag_api.search_products(query, top_k)
        return jsonify({
            "query": query,
            "total_results": len(results),
            "products": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"Error en búsqueda: {str(e)}"}), 500


@app.route('/chat', methods=['GET', 'POST'])
def chat_with_products():
    """Endpoint de chat con contexto de productos"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data or 'query' not in data:
                return jsonify({"error": "Se requiere el campo 'query' en el JSON"}), 400
            query = data['query'].strip()
            top_k = data.get('top_k', 3)
        else:  # GET
            query = request.args.get('query', '').strip()
            top_k = request.args.get('top_k', default=3, type=int)

        if not query:
            return jsonify({"error": "La consulta no puede estar vacía"}), 400
        if not rag_api.is_loaded:
            return jsonify({"error": "API no inicializada correctamente"}), 503

        result = rag_api.chat_with_context(query, top_k)
        return jsonify({
            "query": query,
            "chat_response": result["response"],
            "products_used": result["products_used"],
            "has_error": result["error"],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"Error en chat: {str(e)}"}), 500

@app.route('/stats', methods=['GET'])
def get_catalog_stats():
    """Endpoint de estadísticas del catálogo"""
    try:
        if not rag_api.is_loaded:
            return jsonify({
                "error": "API no inicializada correctamente"
            }), 503
        
        stats = rag_api.get_stats()
        
        return jsonify({
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error obteniendo estadísticas: {str(e)}"
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Endpoint para obtener todas las categorías"""
    try:
        if not rag_api.is_loaded:
            return jsonify({
                "error": "API no inicializada correctamente"
            }), 503
        
        categories = []
        cat_counts = rag_api.df['CATEGORIA'].value_counts()
        
        for cat, count in cat_counts.items():
            categories.append({
                "name": cat,
                "count": int(count)
            })
        
        return jsonify({
            "total_categories": len(categories),
            "categories": categories,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error obteniendo categorías: {str(e)}"
        }), 500

@app.route('/category/<category_name>', methods=['GET'])
def get_products_by_category(category_name):
    """Endpoint para obtener productos por categoría"""
    try:
        if not rag_api.is_loaded:
            return jsonify({
                "error": "API no inicializada correctamente"
            }), 503
        
        limit = request.args.get('limit', 20, type=int)
        
        products = rag_api.get_products_by_category(category_name, limit)
        
        return jsonify({
            "category": category_name,
            "total_results": len(products),
            "products": products,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error obteniendo productos por categoría: {str(e)}"
        }), 500

@app.route('/product/<int:product_code>', methods=['GET'])
def get_product_by_code(product_code):
    """Endpoint para obtener un producto específico por código"""
    try:
        if not rag_api.is_loaded:
            return jsonify({
                "error": "API no inicializada correctamente"
            }), 503
        
        product_row = rag_api.df[rag_api.df['COD'] == product_code]
        
        if len(product_row) == 0:
            return jsonify({
                "error": f"Producto con código {product_code} no encontrado"
            }), 404
        
        row = product_row.iloc[0]
        product = {
            "codigo": int(row['COD']),
            "categoria": row['CATEGORIA'],
            "producto": row['PRODUCTO'],
            "precio": int(row['PRECIO_MILES']),
            "precio_formatted": f"${row['PRECIO_MILES']}k"
        }
        
        return jsonify({
            "product": product,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error obteniendo producto: {str(e)}"
        }), 500

# Endpoint de documentación
@app.route('/docs', methods=['GET'])
def api_documentation():
    """Documentación de la API"""
    docs = {
        "title": "PC Parts RAG API",
        "version": "1.0.0",
        "description": "API para búsqueda inteligente en catálogo de partes de PC",
        "base_url": request.base_url.replace('/docs', ''),
        "endpoints": {
            "GET /health": "Estado de la API",
            "POST /search": "Búsqueda de productos (JSON: {\"query\": \"texto\", \"top_k\": 5})",
            "POST /chat": "Chat con contexto (JSON: {\"query\": \"pregunta\", \"top_k\": 3})",
            "GET /stats": "Estadísticas del catálogo",
            "GET /categories": "Lista todas las categorías",
            "GET /category/<name>": "Productos por categoría",
            "GET /product/<code>": "Producto específico por código",
            "GET /docs": "Esta documentación"
        },
        "example_requests": {
            "search": {
                "url": "/search",
                "method": "POST",
                "body": {"query": "memoria DDR4 8GB", "top_k": 5}
            },
            "chat": {
                "url": "/chat", 
                "method": "POST",
                "body": {"query": "¿Cuál es la memoria más barata?", "top_k": 3}
            }
        }
    }
    
    return jsonify(docs)

def initialize_api():
    """Inicializa la API cargando los datos"""
    print("🚀 Inicializando API de PC Parts...")
    
    if not rag_api.load_system():
        print("❌ Error: No se pudo inicializar la API")
        print("💡 Asegúrate de tener:")
        print("  1. partes_pc.csv en el directorio")
        print("  2. pc_parts_vectors.pkl o pc_parts_vectors.faiss")
        print("  3. Ollama ejecutándose en localhost:11434")
        return False
    
    return True

@app.route('/chat-ui')
def chat_ui():
    return render_template("chat.html")

if __name__ == '__main__':
    # Inicializar API
    if initialize_api():
        print("\n" + "="*50)
        print("🚀 PC PARTS RAG API INICIADA")
        print("="*50)
        print("\n💡 Asegúrate de tener Ollama ejecutándose en localhost:11434")
        print("   y los archivos de vectores en el directorio actual.")
        print("📡 Endpoints disponibles:")
        print("  • http://localhost:5000/health")
        print("  • http://localhost:5000/search (POST)")  
        print("  • http://localhost:5000/chat (POST)")
        print("  • http://localhost:5000/stats")
        print("  • http://localhost:5000/docs")
        print("\n🔍 Ejemplo de uso:")
        print('  curl -X POST http://localhost:5000/search \\')
        print('    -H "Content-Type: application/json" \\')
        print('    -d \'{"query": "memoria DDR4", "top_k": 3}\'')
        print("  • http://localhost:5000/chat-ui")
        print("="*50)
        
        # Iniciar servidor Flask
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ No se pudo iniciar la API")

    