![Precios (2)](https://github.com/user-attachments/assets/23202b7a-4abb-483c-b69e-1ae9ee65092e)


# Descubrimientos del Challenge

En las siguientes lineas voy a describir mi viaje de aprendizaje de redes neuronales, intente ser muy conceptual y entender principalmente los fundamentos, lo cual fue una muy buena decisión y lo recomiendo para cualquiera que quiera aprender esto a profundidad.

## La red neuronal no es algo físico
Al principio me imaginaba algo físico, como un mini procesador que tenía que alojar en algún lado. Fue mi primera sorpresa descubrir que es simplemente código en un notebook. No hay nada que "instalar" o "descargar" más allá de NumPy. La red neuronal existe solo como matrices de números que viven en la memoria de mi computadora mientras el código corre.

## Comprender qué haría la red
El objetivo era clasificar imágenes entre 10 categorías diferentes de ropa usando Fashion MNIST. Es un problema de clasificación multiclase: dada una imagen de 28×28 píxeles, la red debe decidir si es una camiseta, pantalón, vestido, zapatilla, etc. Simple en concepto, complejo en implementación.

## La diferencia fundamental con mis proyectos previos de ML
Aquí es donde realmente entendí el salto que estaba dando. Antes usaba modelos ya construidos como RandomForest o DecisionTree - era usuario de herramientas. Llamaba `.fit()` y `.predict()` sin saber qué matemáticas ocurrían adentro. Ahora construiría cada pieza matemática del modelo - sería constructor de la herramienta. La analogía que me ayudó: antes usaba el auto, ahora construiría el motor.

## El valor real del proyecto
Rápidamente comprendí que el valor no estaba en crear algo innovador. Este es literalmente el "Hello World" de las redes neuronales - miles de personas lo han hecho antes. El valor está en entender las bases y fundamentos que hay detrás. No es glamoroso, pero me prepara para entender PyTorch y TensorFlow después, cuando necesite construir algo realmente complejo.

## Por qué solo NumPy?
Las indicaciones del proyecto tenían esta limitación, luego entendí su propósito. Me obliga a programar las matemáticas manualmente, me hace entender cada operación que hace la red. Si usara TensorFlow o PyTorch, estaría usando "cajas negras" nuevamente. Necesitaba ver qué pasa por dentro, aunque fuera más trabajoso.

## Elección de Fashion MNIST
Elegí Fashion MNIST sobre MNIST regular. Técnicamente son idénticos (mismas dimensiones, mismo formato), pero Fashion MNIST es más motivante e interesante visualmente. Ver camisetas y zapatos en lugar de dígitos conectaba mejor con la narrativa del challenge. Pequeños detalles que mantienen la motivación alta.

## Entendiendo la estructura de datos
Mi primer obstáculo real: Fashion MNIST no era un DataFrame de pandas como en proyectos anteriores. Eran arrays de NumPy, una estructura multidimensional que no dominaba. Una imagen es una matriz 28×28 con valores entre 0-255 representando intensidad de píxeles. Las 60,000 imágenes forman un "stack" de matrices. La analogía del libro con 60,000 páginas me ayudó: cada página es una imagen, las etiquetas son post-its pegados indicando qué representa cada imagen.

## Preparación de datos
Tres pasos resultaron ser más directos de lo que anticipaba. Normalización: dividir entre 255 para llevar valores al rango 0-1 (evita que números enormes desestabilicen el entrenamiento). Flatten: convertir cada matriz 28×28 en un vector de 784 números. One-hot encoding: convertir un número como 9 en un vector [0,0,0,0,0,0,0,0,0,1]. Todo con operaciones NumPy eficientes, sin necesidad de loops.

## Broadcasting: la magia invisible de NumPy
Descubrí algo fascinante: broadcasting. NumPy expande automáticamente arrays pequeños para que coincidan con grandes. Cuando sumo una matriz (60000, 10) con un vector (10,), NumPy replica el vector 60,000 veces automáticamente. Hace el código más limpio y 100x más rápido que usar loops manuales. El parámetro `keepdims=True` resultó crucial para controlar estas expansiones correctamente.

## La arquitectura es una decisión de diseño
Algunas cosas eran fijas: 784 neuronas en la entrada (tamaño de la imagen aplanada) y 10 neuronas en la salida (las 10 categorías). Pero las neuronas en la capa oculta eran un hiperparámetro que YO decidía. No había respuesta "correcta". Elegí empezar con solo 10 neuronas para experimentar, sabiendo que probablemente no sería suficiente. La idea era ver el efecto de aumentarlas después.

## Matemáticas
Hice muchas de las matemáticas manualmente a menor escala, esto me permitía entender más a profundidad cómo configurar correctamente el tamaño de las matrices y entender conceptualmente. Por ejemplo al tener una multiplicación de A1 x W2, comprender que estaba multiplicando esencialmente pesos de neuronas.

## Funciones de activación: por qué son necesarias
Aprendí que sin funciones de activación, múltiples capas colapsarían matemáticamente en una sola operación lineal - completamente inútil. ReLU en capas ocultas introduce no-linealidad, permitiendo aprender patrones complejos. Su definición es ridículamente simple: ReLU(x) = max(0, x). Softmax en la salida convierte números arbitrarios en probabilidades que suman 1.0. No es magia, es matemáticas elegante.

## Forward pass: el primer momento revelador
Implementé la predicción completa y la ejecuté por primera vez, de manera manual sin loops. La red con pesos aleatorios predijo con ~10% de accuracy - puro azar, como esperaba. Pero ver que funcionaba matemáticamente, aunque las predicciones fueran malas, fue increíblemente satisfactorio. Confirmaba que la estructura estaba correcta antes de intentar entrenar. Era como ver el motor arrancar por primera vez, aunque no fuera a ningún lado todavía.

![Precios (1)](https://github.com/user-attachments/assets/7173f4ca-3370-4319-935f-0ff8352446d4)


<img width="3305" height="472" alt="image" src="https://github.com/user-attachments/assets/37eba104-12aa-4baf-9024-0e9d228a6083" />




## Loss y Accuracy: dos formas de medir
Necesitaba dos métricas diferentes. Loss (Cross-Entropy) mide qué tan "insegura" está la red usando -log(probabilidad_correcta). Penaliza mucho más estar muy equivocado que estar un poco equivocado. Accuracy es simplemente el porcentaje de predicciones correctas - más intuitivo pero menos informativo para el entrenamiento. Entender la diferencia fue clave: loss guía el entrenamiento, accuracy la entiendo yo.

## Backpropagation: el concepto vs la implementación
Llegué al momento más intimidante: backpropagation. El concepto es calcular cómo ajustar cada peso para reducir el error, usando la regla de la cadena del cálculo. Podría haber intentado derivar las fórmulas manualmente, pero tomé una decisión pragmática: usar las fórmulas finales directamente. Lo importante era entender QUÉ hace cada fórmula, no demostrar teoremas matemáticos. Estoy aprendiendo ingeniería, no matemática pura.

![Precios](https://github.com/user-attachments/assets/a692e7b7-d545-4809-b106-113c2be18e62)


<img width="3566" height="438" alt="image" src="https://github.com/user-attachments/assets/0f6cebc8-6ca4-4d7f-a3f9-c25903879460" />


## El primer entrenamiento: magia pura
Ejecuté el loop de entrenamiento por primera vez. Una iteración: accuracy subió de 6.6% a 10%. Cien iteraciones: llegó a 55%. La red APRENDIÓ sin que yo le dijera explícitamente cómo clasificar camisetas o zapatos. Solo con matemáticas - ajustando números automáticamente basándose en errores. Este fue el momento "wow" del proyecto. Entender la teoría es una cosa, ver los números mejorando epoch tras epoch es otra completamente diferente.

## Experimentación sistemática
Me di cuenta de que necesitaba ser metódico. Creé un sistema para guardar resultados de cada experimento: neuronas, epochs, accuracy en train y test, loss, tiempo de entrenamiento. Comparé arquitecturas diferentes de forma controlada, cambiando una variable a la vez. Este enfoque científico me permitió entender qué mejoras realmente funcionaban versus cuáles eran solo ruido. Documentar experimentos resultó ser tan importante como programarlos.

## Más neuronas ≠ siempre mejor
Los números contaron una historia interesante. Pasar de 10 a 64 neuronas dio +5.9% de accuracy - gran mejora. Pero de 64 a 128 neuronas solo +1.8% - rendimientos decrecientes. Duplicar neuronas también duplicaba el tiempo de entrenamiento. Hay un balance entre capacidad del modelo y eficiencia. Para este problema, 128 neuronas encontraron un buen equilibrio. Más no necesariamente significaba mejor.

## Epochs: paciencia vs resultados
Otro descubrimiento: la red seguía mejorando con más entrenamiento. 100 epochs alcanzaron 66% accuracy. 200 epochs llegaron a 71%. 400 epochs alcanzaron 79%. La mejora era continua sin señales de overfitting. Más tiempo de entrenamiento SÍ valía la pena cuando la red tenía suficiente capacidad. La paciencia literalmente se tradujo en mejores resultados. Cada epoch tomaba ~40 segundos, así que 400 epochs significaban esperar ~4.5 horas. Valió la pena.

## No hay overfitting en mi red
Algo curioso: el "problema" de overfitting que debía enfrentar según el challenge nunca apareció. En todos mis experimentos, train accuracy ≈ test accuracy (diferencia <2%). Esto indicaba buena generalización. La red era suficientemente simple - una sola capa oculta con regularización implícita por su simplicidad. Irónicamente, "mostrar evidencia de cómo enfrenté el overfitting" significó mostrar que monitoreé train vs test y que mi arquitectura simple lo previno naturalmente.

## Refactorización: cuando el código pide orden
Después de hacer funcionar todo el pipeline de entrenamiento, el código pedía a gritos organización. Lo refactoricé en funciones limpias: `initialize_parameters`, `forward_pass`, `backward_pass`, `compute_loss`, `train_loop`, `evaluate`. No lo hice antes - primero necesitaba entender, luego organizar. Este orden resultó correcto. Las funciones hicieron que ejecutar nuevos experimentos fuera trivial: cambiar parámetros y correr. El código se convirtió en una herramienta en lugar de un experimento.

## El resultado final
Alcancé ~79% de test accuracy con la arquitectura 784→128→10 en 400 epochs. Lejos del 95% "perfecto" que algunos logran con redes más complejas, pero cumplía completamente el objetivo educativo. Más importante: entiendo cada línea de código que escribí. Puedo explicar cada operación matemática. Sé exactamente qué pasa cuando los datos fluyen por la red. ESE es el verdadero logro del challenge. No construí la red neuronal más precisa, construí mi comprensión de cómo funcionan las redes neuronales.
