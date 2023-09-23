import torch
import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib as plt
import torchinfo, torchmetrics

# Scalar
# Het kan een geheel getal, een breuk, of een reëel getal zijn. 
# In wiskundige en programmeertaalcontexten wordt de term "scalar" vaak gebruikt om een enkel numeriek waarde te beschrijven die geen richting of meerdere dimensies heeft,
# in tegenstelling tot vectoren of matrices. 
# Scalars worden bijvoorbeeld gebruikt om vectoren en matrices te schalen of om een grootte (magnitude) weer te geven.

scalar = torch.tensor(1)
scalar.item() # Dit is corresponderend aan print(scalar)

# Vector
# Een vector vaak weergegeven als een eendimensionale array

# Tensor
# Een tensor is eigenlijk een soort algemene term voor een verzameling getallen die in een specifieke vorm zijn gerangschikt. 
# Je kunt het zien als een uitbreiding van een gewone getallenlijst (vector) of een tabel van getallen (matrix), maar dan in meerdere dimensies. 
# In de context van programmeren en machine learning worden tensoren gebruikt om data op te slaan en te manipuleren.

TENSOR = torch.tensor([[[[2, 4, 6], [8, 10, 12], [14, 16, 18]]]])
arange = torch.arange(start=0, end=100, step=10) # dit output tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
arange_zeros = torch.zeros_like(input=arange) # dit output tensor([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# De drie meest voorkomende errors die je krijgt in Pytorch & deep learning:
# 1. Tensors not right datatype (e.g. int16, int32, int64)
# 2. Tensors not right shape (inner dimensions, outer dimensions)
# 3. Tensors not on the right device

#--------------------------------- Tensor datatypes ----------------------------------------#

float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                               dtype=None, # Wat voor datatype is het? Bijvoorbeeld: float16, float32 of float64
                               device=None, # Welk device gebruik je? Bijvoorbeeld: CPU of GPU
                               requires_grad=False) # Of je wel of niet gradients moet tracken

random_tensor = torch.tensor([1, 2])
random_tensor_data_type = random_tensor.dtype
random_tensor_shape = random_tensor.shape
random_tensor_device = random_tensor.device

#--------------------------------- Manipulating tensor ----------------------------------------#

# Manipulating tensors (Element wise)
# Addition (+)
# Subtraction = (-)
# Multiplication (*)
# Division (/)

# Matrix multiplication (dot(•) product) https://www.mathsisfun.com/algebra/matrix-multiplying.html
# (1, 2, 3) • (8, 10, 12) = 1×8 + 2×10 + 3×12 = 64
# In dit geval is random_tensor (1, 2) dus is het 1x1 (= 1) + 2x2 (= 4) dus is het getal 5
# Je kan ook @ gebruiken, dus bijvoorbeeld: random_tensor @ random_tensor. Dit zal ook 5 returnen. @ staat dus voor matmul (Matrix multiplication)

matrix_multi = torch.matmul(random_tensor, random_tensor)

# Matrix multiplaction heeft twee regels waar je aan moet voldoen.
# Regel 1
# De "inner dimensions" moeten overeenkomen voor matrixvermenigvuldiging. In het geval van torch.matmul(torch.rand(1, 2), torch.rand(1, 2)), heb je twee matrices van vorm 1×2.
# De "inner dimensions" (het tweede getal in 1x2) is 2, en de "inner dimensions" van de tweede matrix (het eerste getal in 1x2) is 1. Dus komen deze NIET overheen

matrix_multi_error = torch.matmul(torch.tensor([1, 2]), torch.tensor([1, 2])) # Dit werkt NIET
# Dit geeft de error: RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x2 and 1x2)

matrix_multi_correct = torch.matmul(torch.tensor([2, 3]), torch.tensor([3, 2])) # Dit werkt
matrix_multi_correct = torch.matmul(torch.tensor([3, 2]), torch.tensor([2, 3])) # Dit werkt

# Zie je hoe de binnenste getallen beide het zelfde zijn, dus van de eerste 2,3 is 3 het binneste getal, en 3,2 is 3 ook het binnenste getal.

# Regel 2
# Het resultaat van de matrix heeft de shape van de outer dimmension. Dus in het geval van torch.matmul(torch.rand(3, 2), torch.rand(2, 3)). Heeft het een shape van 3.
# Omdat de "outer dimensions" 3 is (het eerste getal in 3x2), en de "outer dimensions" van de tweede matrix (het tweede getal in 2x3) beide 3 zijn, is de shape dus 3.

#--------------------------------- Transpose ----------------------------------------#

tensor_a = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_b = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])

# In dit voorbeeld hebben de tensors tensor_a en tensor_b beide de vorm 3×2. Als je matrixvermenigvuldiging wilt uitvoeren, moeten de "inner dimensions" overeenkomen.
# In dit geval zou het tweede getal van de eerste matrix gelijk moeten zijn aan het eerste getal van de tweede matrix. Aangezien beide matrices de vorm 3×2
# hebben, komen de "inner dimensions" niet overeen voor een standaard matrixvermenigvuldiging.
# Om dit probleem op te lossen, kun je een van de matrices transponeren. De getransponeerde matrix zal dan de vorm 2×3 hebben, waardoor de "inner dimensions" wel overeenkomen.

result = torch.matmul(tensor_a, tensor_b.T)

# De .T in PyTorch is een eigenschap die de getransponeerde van een tensor teruggeeft. 
# Transponeren is het proces waarbij je de rijen en kolommen van een matrix omwisselt. Dus, bijvoorbeeld:

# Dit is jou matrix:
# 1 & 2
# 3 & 4 
# 5 & 6
# Als je hem transponeert
# 1 & 3 & 5
# 2 & 4 & 6

# Hierdoor krijgt tensor_b dus een andere shape, namelijk: 3x2. Hierdoor zal de "inner dimensions" wel overheenkomen met elkaar en kun je daardoor matrixvermenigvuldiging toepassen
# De shape van de matrix ligt aan de "outer dimensions". Dus stel je doet [3, 2] @ [2, 3] is de shape 3 x 3. En bij [2, 3] @ [3, 2] is de shape 2 x 2.

result = torch.matmul(tensor_a, tensor_b.T) # Shape zal zijn: 3 x 3 ([3, 2] @ [2, 3])
result = torch.matmul(tensor_a.T, tensor_b) # Shape zal zijn: 2 x 2 ([2, 3] @ [3, 2])

#--------------------------------- Tensor aggregation ----------------------------------------#

# Tensor aggregation (Zoeken van min, max, mean, sum)

# Laten we een nieuwe tensor maken doormiddel van arange.

arangeTensor = torch.arange(0, 110, 10) # start, einde en het aantal stappen.

# Vind de min (het laagste getal)
arangeTensor.min() # of torch.min(arangeTensor)

# Vind de max (het hoogste getal)
arangeTensor.max()

# Vind de mean. Standaard is de datatype int64, die niet wordt toegelaten door de mean functie, dus moet je de datatype aanpassen van de tensor.
# Anders krijg je deze error: RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
# Hij geeft al aan dat het een floating point of een complex dtype moet zijn.
arangeTensor.type(torch.float32).mean()

# Vind de sum (Alles bij elkaar opgeteld)
arangeTensor.sum()

# Vind de positie in tensor van de minimum value -> returned de index van deze minimum value
arangeTensor.argmin()

# Vind de positie in tensor van de maximum value -> returned de index van deze maximum value
arangeTensor.argmax()

#--------------------------------- Reshaping, stacking, squeezing and unsqueezing tensors ----------------------------------------#

# Reshaping - Verandert de vorm van een invoertensor naar een gedefinieerde vorm
# View - Geeft een weergave terug van een invoertensor met een bepaalde vorm, maar behoudt hetzelfde geheugen als de originele tensor
# Stacking - Combineert meerdere tensors bovenop elkaar (vstack) of naast elkaar (hstack)
# Squeeze - Verwijdert alle `1` dimensies uit een tensor
# Unsqueeze - Voegt een `1` dimensie toe aan een doeltensor
# Permute - Geeft een weergave terug van de invoer met dimensies die op een bepaalde manier zijn verwisseld (omgewisseld)

# Create new tensor
x = torch.arange(1, 10) # dit returned: tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]) torch.Size([9])

# Voeg een extra dimensie toe
x_reshaped = x.reshape(1, 9) # De reshape moet overheen komen het de size. Dus 1 en 9 kan, omdat 1 maal 9 = 9. Maar 2 en 9 kan niet. Want 2 maal 9 is 18.
# x.reshape(1, 9) zal de tensor omvormen tot een 2-dimensionale tensor met 1 rij en 9 kolommen, dus als volgt: [[1, 2, 3, 4, 5, 6, 7, 8, 9]].
# x.reshape(9, 1) zal de tensor omvormen tot een 2-dimensionale tensor met 9 rijen en 1 kolom, dus als volgt:
# tensor([[1],
#        [2],
#        [3],
#        [4],
#        [5],
#        [6],
#        [7],
#        [8],
#        [9]])
# Beide operaties behouden het aantal elementen (9 in dit geval), maar de dimensies en de vorm van de tensor zijn anders. 
# Het eerste resultaat is een rijvector, en het tweede resultaat is een kolomvector.

# Verander de view
z = x.view(1, 9) # Dit returned tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]) torch.Size([1, 9])
# Het veranderen van Z zal ook X veranderen. (Want de view van een tensor deelt dezelfde memory als de orginele memory)

# Stack tensors op elkaar
x_stacked = torch.stack([x, x, x, x], dim=0) # Dit returned 
# tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
#        [1, 2, 3, 4, 5, 6, 7, 8, 9],
#        [1, 2, 3, 4, 5, 6, 7, 8, 9],
#        [1, 2, 3, 4, 5, 6, 7, 8, 9]])

# Bij `dim=2` krijg je een foutmelding. Dit komt omdat de invoertensors 1-dimensionaal is en er geen 2e dimensie bestaat om langs te stapelen. 
# De maximale waarde voor dim is gelijk aan het aantal dimensies van de tensor (in dit geval is dat 1).
# Om stapeling langs de 2e dimensie mogelijk te maken, zou je de invoertensors eerst moeten omvormen tot ten minste 2-dimensionale tensors.

# torch.squeeze() - Verwijderd alle single dimensions van een tensor
x_squeezed = x_reshaped.squeeze() # Dit maakt van tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]) naar tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]). Hij haalt simpel weg gewoon de dimension weg.

# toch.unsqueeze() - Voegt een single dimension toe aan een tensor naar een specifieke dim (dimension)
x_unsqueezed = x_squeezed.unsqueeze(dim=0) # Dim pakt de index van de size, dus (1, 9) is de 1 een 0 want python begint met tellen vanaf 0's. 
# Stel ik doe dim=1 dan krijg ik dezelfde shape als `x.reshape(9, 1)` bij regel 157
# Dus dit maakt de squeezed tensor `tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])` weer een `tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])`

x_original = torch.rand(size=(224, 275, 3)) # returned torch.Size([224, 224, 3])
x_premute = torch.permute(x_original, (2, 0, 1)).size() # 2, 0, 1 zijn de indexes. Dus je ziet 2 = 3, 0 = 224 en 1 = 275. Omdat python bij 0 begint met tellen.
# x_premute returned torch.Size([3, 224, 275])

# Indexing (Selecteer data van tensors)

# Create new tensor
m = torch.arange(1, 10).reshape(1, 3, 3) # Reshape moet gelijk zijn aan de size, want 1 ⋅ 3 ⋅ 3 = 9
# Dit returned: 
# tensor([[[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]]])
# De `1` staat voor 1 dimension. Dus de [] brackets
# De eerste `3` staat voor 3 columns ([1, 2, 3] = 1 column)
# De tweede `3` staat voor 3 per column ([1, 2, 3] heeft 3 values)

m # returned [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
m[0] # returned [[1, 2, 3], [4, 5, 6], [7, 8, 9]]. Hij gaat 1 dimension erin.
m[0][0] # returned [1, 2, 3]
m[0][0][0] # returned 1
m[0][0][2] # returned 3
m[0, 0, 2] # Kan ook

# Zo ga je steeds dieper de tensor in
# Je kan ook ":" gebruiken om "alles" te targeten van de dimension
m[:, 0] # returned [1, 2, 3]. Want de eerste `:` pakt alles, dus dat is simpel weg dimension 1. En 0 is dan [1, 2 ,3]. Dus de eerste index
m[:, 1, 1] # dit returned `5`. Het is in dit geval het zelfde als [0, 1, 1]
m[:, :, 1] # returned [2, 5, 8]. Omdat deze alle 1e indexes pakt van de columns. Door die middelste te veranderen naar `:` pak je alle `1` indexes.