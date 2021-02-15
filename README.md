# Segundo Proyecto (Espacios vectoriales complejos)

Libreria con operaciones basicas de matrices complejas y reales

### Prerequisitos
Pycharm instalado en su dispositivo

## Correr los tests
Para probar la libreria usando los test unitarios se debe buscar la operacion
deseada y asignarle un valor a las respectivas variables, de la manera
que se muestra en el siguiente ejemplo dv
```
  def test_sum_mat(self):
      mat1 = np.array([[1+2j], [5+1j], [7+8j]])
      mat2 = np.array([[4+3j], [1+2j], [3+2j]])
      res = np.array([[5.+5.j], [6.+3.j], [10.+10.j]])
      self.assertTrue(
          np.testing.assert_almost_equal(esv.mat_sum(mat1, mat2), res) is None)
  
```
En este ejemplo (suma de matrices), mat1 y mat2 son las matrices a sumar, res
es el resultado esperado, usted tendra que modificar estos valores a los que
desee

## Hecho con

* [PyCharm](https://www.jetbrains.com/es-es/pycharm/) - IDE usada

# Autor
* **Miguel Angel Salamanca**