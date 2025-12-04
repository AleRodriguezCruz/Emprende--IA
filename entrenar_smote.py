import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("📊 ENTRENAMIENTO CON GRÁFICAS - MODELO BALANCEADO SMOTE")

# ===========================================
# 1. CARGAR Y ANALIZAR DATOS
# ===========================================

print("\n" + "="*50)
print("1. ANÁLISIS DE DATOS")
print("="*50)

# Cargar datasets
df_original = pd.read_csv('oportunidades_completo_v5.csv')
df_smote = pd.read_csv('oportunidades_balanceadas_smote.csv')

print(f"📁 Dataset original: {df_original.shape}")
print(f"📁 Dataset SMOTE: {df_smote.shape}")

# Features
FEATURES = [
    'densidad_escuelas', 'densidad_hospitales', 'densidad_gimnasios',
    'densidad_papelerias', 'densidad_farmacias', 'densidad_comercios'
]

# ===========================================
# GRÁFICA 1: Distribución antes/después de balanceo
# ===========================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Antes de balanceo
dist_original = df_original['target'].value_counts().sort_index()
axes[0].bar(dist_original.index.astype(str), dist_original.values, color='salmon', alpha=0.7)
axes[0].set_title('Distribución Original (Desequilibrada)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Clase')
axes[0].set_ylabel('Número de muestras')
axes[0].grid(True, alpha=0.3)

# Añadir valores en las barras
for i, v in enumerate(dist_original.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Después de balanceo
dist_smote = df_smote['target'].value_counts().sort_index()
axes[1].bar(dist_smote.index.astype(str), dist_smote.values, color='lightgreen', alpha=0.7)
axes[1].set_title('Distribución después de SMOTE (Balanceada)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Clase')
axes[1].set_ylabel('Número de muestras')
axes[1].grid(True, alpha=0.3)

for i, v in enumerate(dist_smote.values):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('1_distribucion_balanceo.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica 1 guardada: 1_distribucion_balanceo.png")

# ===========================================
# 2. PREPARAR DATOS PARA ENTRENAMIENTO
# ===========================================

print("\n" + "="*50)
print("2. PREPARACIÓN DE DATOS")
print("="*50)

# Usar dataset SMOTE balanceado
df = df_smote
X = df[FEATURES]
y = df['target']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔧 Conjunto de entrenamiento: {X_train.shape}")
print(f"🎯 Conjunto de prueba: {X_test.shape}")

# Mapeo de clases
clases_df = df[['oportunidad_negocio', 'target']].drop_duplicates().sort_values('target')
clases_dict = dict(zip(clases_df['target'], clases_df['oportunidad_negocio']))

print(f"\n🎯 CLASES DEL MODELO:")
for codigo, nombre in clases_dict.items():
    print(f"   Clase {codigo}: {nombre}")

# ===========================================
# 3. ENTRENAR MODELO
# ===========================================

print("\n" + "="*50)
print("3. ENTRENAMIENTO DEL MODELO")
print("="*50)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("🎯 Entrenando Random Forest...")
model.fit(X_train, y_train)
print("✅ Modelo entrenado exitosamente")

# ===========================================
# 4. EVALUACIÓN Y GRÁFICAS
# ===========================================

print("\n" + "="*50)
print("4. EVALUACIÓN DEL MODELO")
print("="*50)

# Predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy total: {accuracy:.4f} ({accuracy:.2%})")

# Reporte de clasificación
print("\n📋 REPORTE DE CLASIFICACIÓN:")
print(classification_report(
    y_test, y_pred,
    target_names=[clases_dict[i] for i in sorted(clases_dict.keys())]
))

# ===========================================
# GRÁFICA 2: Matriz de confusión
# ===========================================

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[clases_dict[i] for i in sorted(clases_dict.keys())],
            yticklabels=[clases_dict[i] for i in sorted(clases_dict.keys())])
plt.title('Matriz de Confusión - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('2_matriz_confusion.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica 2 guardada: 2_matriz_confusion.png")

# ===========================================
# GRÁFICA 3: Importancia de características
# ===========================================

plt.figure(figsize=(10, 6))
importancias = model.feature_importances_
indices = np.argsort(importancias)[::-1]

plt.bar(range(len(importancias)), importancias[indices])
plt.xticks(range(len(importancias)), [FEATURES[i] for i in indices], rotation=45, ha='right')
plt.title('Importancia de Características - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3_importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica 3 guardada: 3_importancia_caracteristicas.png")

# ===========================================
# GRÁFICA 4: Curvas ROC por clase
# ===========================================

plt.figure(figsize=(10, 8))

# Binarizar las etiquetas para ROC multiclase
y_test_bin = label_binarize(y_test, classes=sorted(clases_dict.keys()))
n_classes = y_test_bin.shape[1]

# Calcular ROC para cada clase
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2,
             label=f'Clase {sorted(clases_dict.keys())[i]} ({clases_dict[sorted(clases_dict.keys())[i]]}) - AUC = {roc_auc:.3f}')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC por Clase', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4_curvas_roc.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica 4 guardada: 4_curvas_roc.png")

# ===========================================
# GRÁFICA 5: Accuracy por clase
# ===========================================

plt.figure(figsize=(10, 6))
report = classification_report(y_test, y_pred, output_dict=True)
clases_nombres = [clases_dict[i] for i in sorted(clases_dict.keys())]
precisions = [report[str(i)]['precision'] for i in sorted(clases_dict.keys())]
recalls = [report[str(i)]['recall'] for i in sorted(clases_dict.keys())]
f1_scores = [report[str(i)]['f1-score'] for i in sorted(clases_dict.keys())]

x = np.arange(len(clases_nombres))
width = 0.25

plt.bar(x - width, precisions, width, label='Precisión', alpha=0.8)
plt.bar(x, recalls, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

plt.xlabel('Clases')
plt.ylabel('Puntuación')
plt.title('Métricas por Clase', fontsize=14, fontweight='bold')
plt.xticks(x, clases_nombres, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('5_metricas_por_clase.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica 5 guardada: 5_metricas_por_clase.png")

# ===========================================
# 5. GUARDAR MODELO Y RESULTADOS
# ===========================================

print("\n" + "="*50)
print("5. GUARDAR RESULTADOS")
print("="*50)

# Guardar modelo
model_filename = 'modelo_oportunidades_smote.pkl'
clases_filename = 'clases_oportunidades_smote.pkl'

joblib.dump(model, model_filename)
joblib.dump(clases_df, clases_filename)

print(f"💾 Modelo guardado: {model_filename}")
print(f"💾 Clases guardadas: {clases_filename}")

# Guardar reporte en texto
with open('reporte_modelo.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("REPORTE DEL MODELO RANDOM FOREST\n")
    f.write("="*60 + "\n\n")
    
    f.write("DATOS:\n")
    f.write(f"- Dataset original: {df_original.shape}\n")
    f.write(f"- Dataset SMOTE: {df_smote.shape}\n")
    f.write(f"- Entrenamiento: {X_train.shape}\n")
    f.write(f"- Prueba: {X_test.shape}\n\n")
    
    f.write("DISTRIBUCIÓN DE CLASES (SMOTE):\n")
    for clase, conteo in dist_smote.items():
        f.write(f"- Clase {clase} ({clases_dict[clase]}): {conteo} muestras\n")
    f.write("\n")
    
    f.write(f"ACCURACY TOTAL: {accuracy:.4f} ({accuracy:.2%})\n\n")
    
    f.write("REPORTE DE CLASIFICACIÓN:\n")
    f.write(classification_report(
        y_test, y_pred,
        target_names=[clases_dict[i] for i in sorted(clases_dict.keys())]
    ))
    
    f.write("\n" + "="*60 + "\n")
    f.write("IMPORTANCIA DE CARACTERÍSTICAS:\n")
    for i in indices:
        f.write(f"- {FEATURES[i]}: {importancias[i]:.4f}\n")

print("📝 Reporte guardado: reporte_modelo.txt")

# ===========================================
# 6. PREDICCIONES DE EJEMPLO (CORREGIDO)
# ===========================================

print("\n" + "="*50)
print("6. PREDICCIONES DE EJEMPLO")
print("="*50)

def predecir_ejemplo(features_dict):
    features = np.array([[features_dict[f] for f in FEATURES]])
    prediccion = model.predict(features)[0]
    probabilidades = model.predict_proba(features)[0]
    
    # Crear diccionario de probabilidades SOLO para clases existentes
    prob_dict = {}
    for i, prob in enumerate(probabilidades):
        if i in clases_dict:  # Solo incluir clases que existen
            prob_dict[clases_dict[i]] = float(prob)
        else:
            prob_dict[f"Clase_{i}"] = float(prob)  # O usar nombre genérico
    
    return {
        'oportunidad': clases_dict[prediccion],
        'codigo': int(prediccion),
        'confianza': float(max(probabilidades)),
        'todas_probabilidades': prob_dict
    }

# Ejemplos representativos
ejemplos = [
    {'densidad_escuelas': 3, 'densidad_hospitales': 0, 'densidad_gimnasios': 0,
     'densidad_papelerias': 0, 'densidad_farmacias': 0, 'densidad_comercios': 2},
    
    {'densidad_escuelas': 0, 'densidad_hospitales': 3, 'densidad_gimnasios': 0,
     'densidad_papelerias': 0, 'densidad_farmacias': 1, 'densidad_comercios': 1},
    
    {'densidad_escuelas': 1, 'densidad_hospitales': 1, 'densidad_gimnasios': 2,
     'densidad_papelerias': 0, 'densidad_farmacias': 0, 'densidad_comercios': 5},
]

print("\n🔮 EJEMPLOS DE PREDICCIÓN:")
for i, ejemplo in enumerate(ejemplos, 1):
    resultado = predecir_ejemplo(ejemplo)
    print(f"\nEjemplo {i}:")
    print(f"  Características: {ejemplo}")
    print(f"  → Oportunidad recomendada: {resultado['oportunidad']}")
    print(f"  → Confianza: {resultado['confianza']:.1%}")
    print(f"  → Código: {resultado['codigo']}")
    
    # Mostrar top 3 probabilidades
    top3 = sorted(resultado['todas_probabilidades'].items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  → Top 3 oportunidades:")
    for op, prob in top3:
        print(f"     - {op}: {prob:.1%}")

# ===========================================
# GRÁFICA FINAL: Resumen de resultados
# ===========================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subgráfica 1: F1-Score por clase
axes[0, 0].bar(clases_nombres, f1_scores, color='skyblue', alpha=0.7)
axes[0, 0].set_title('F1-Score por Clase', fontweight='bold')
axes[0, 0].set_ylabel('F1-Score')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Subgráfica 2: Importancia de características
axes[0, 1].barh([FEATURES[i] for i in indices], importancias[indices], color='lightgreen')
axes[0, 1].set_title('Importancia de Características', fontweight='bold')
axes[0, 1].set_xlabel('Importancia')
axes[0, 1].grid(True, alpha=0.3)

# Subgráfica 3: Distribución balanceo
axes[1, 0].bar(['Original', 'SMOTE'], [len(df_original), len(df_smote)], 
               color=['salmon', 'lightgreen'], alpha=0.7)
axes[1, 0].set_title('Tamaño del Dataset', fontweight='bold')
axes[1, 0].set_ylabel('Número de muestras')
axes[1, 0].grid(True, alpha=0.3)

# Subgráfica 4: Accuracy total
axes[1, 1].bar(['Accuracy'], [accuracy], color='gold', alpha=0.7)
axes[1, 1].set_title(f'Accuracy Total: {accuracy:.2%}', fontweight='bold')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].text(0, accuracy/2, f'{accuracy:.2%}', ha='center', va='center', 
                fontsize=20, fontweight='bold', color='white')

plt.suptitle('RESUMEN DEL MODELO - SISTEMA DE OPORTUNIDADES DE NEGOCIO', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('6_resumen_modelo.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfica 6 guardada: 6_resumen_modelo.png")

print("\n" + "="*50)
print("✅ PROCESO COMPLETADO EXITOSAMENTE!")
print("="*50)
print("\n📁 ARCHIVOS GENERADOS:")
print("   1. 1_distribucion_balanceo.png")
print("   2. 2_matriz_confusion.png")
print("   3. 3_importancia_caracteristicas.png")
print("   4. 4_curvas_roc.png")
print("   5. 5_metricas_por_clase.png")
print("   6. 6_resumen_modelo.png")
print("   7. reporte_modelo.txt")
print("   8. modelo_oportunidades_smote.pkl")
print("   9. clases_oportunidades_smote.pkl")
print("\n🎯 ¡Listo para usar en tu API y presentación!")