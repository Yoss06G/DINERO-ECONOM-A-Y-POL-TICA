import pandas as pd
import powerlaw
import matplotlib.pyplot as plt

# Cargar los datos de la encuesta.
# Se asume que el archivo 'enemdu.csv' se encuentra en la ruta especificada y tiene una columna llamada 'ingpc'
data = pd.read_csv(r"d:/SGUERRA330/escritorio/Powerlaw/enemdu.csv", delimiter=";")
print(data.head())
ingresos = data['ingpc']

# Convertir los datos a valores numéricos, forzando errores a NaN
ingresos = pd.to_numeric(ingresos, errors='coerce')

# Limpiar los datos: eliminar valores nulos y considerar solo ingresos positivos
ingresos = ingresos.dropna()
ingresos = ingresos[ingresos > 0]

# Ajustar los datos a una ley de potencia
results = powerlaw.Fit(ingresos)
print("Alpha (exponente):", results.power_law.alpha)
print("xmin:", results.power_law.xmin)


# --- Paso 2: Goodness-of-fit ---
# Usamos el p-value obtenido al comparar con un modelo alternativo (por ejemplo lognormal).
# Si p > 0.1, la hipótesis de la ley de potencia es plausible.
R, p = results.distribution_compare('power_law', 'lognormal')
print("\nGoodness-of-Fit:")
print("Likelihood ratio (R) vs. lognormal:", R)
print("p-value:", p)
if p > 0.1:
    print("El modelo de ley de potencia es plausible (p > 0.1).")
else:
    print("El modelo de ley de potencia es rechazado (p <= 0.1).")


# --- Paso 3: Comparación con hipótesis alternativas ---
alternativas = ['lognormal', 'exponential']
for alt in alternativas:
    R_alt, p_alt = results.distribution_compare('power_law', alt)
    print(f"\nComparación con {alt}:")
    print(f"Likelihood ratio (R): {R_alt}, p-value: {p_alt}")
    if p_alt < 0.1:
        if R_alt < 0:
            print(f"{alt} es favorecido sobre la ley de potencia.")
        elif R_alt > 0:
            print(f"La ley de potencia es favorecida sobre {alt}.")
        else:
            print("No hay preferencia significativa.")
    else:
        print(f"No se obtiene preferencia significativa entre la ley de potencia y {alt}.")


# Graficar la función de densidad de probabilidad en escala log-log
ax = results.plot_pdf(color="b", label="Datos empíricos")
results.power_law.plot_pdf(color="r", linestyle="--", ax=ax, label="Ajuste Power Law")
plt.xlabel("Ingreso por hogar")
plt.ylabel("Probabilidad")
plt.legend()
plt.show()