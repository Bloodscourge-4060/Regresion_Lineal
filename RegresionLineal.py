import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_dataset.csv')

df = pd.DataFrame(data, columns=['YearsExperience', 'Salary'])

x_axis = np.array(data['YearsExperience'])
y_axis = np.array(data['Salary'])

mean_X = np.mean(x_axis)
mean_Y = np.mean(y_axis)


numerador = np.sum((x_axis - mean_X) * (y_axis - mean_Y))
denominador = np.sum((x_axis - mean_X)**2)

m = numerador/denominador
b = mean_Y - (m*mean_X)

print(f"Pendiente (m): {m}")
print(f"Ordenada al origen (b): {b}")

y_pred = m*x_axis+b

mse = np.mean((y_pred - y_axis)**2)
print(f"\n\n Error Cuadratico Medio (MSE): {mse}")

plt.scatter(x_axis, y_axis, color="blue", label="Datos Reales")
plt.plot(x_axis, y_pred, color="red", label="Regresion Lineal")
plt.xlabel("Años de Experiencia")
plt.ylabel("Salario")
plt.legend()
plt.show()