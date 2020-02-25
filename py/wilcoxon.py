from scipy.stats import wilcoxon

dataNB = [0.57954545, 0.83908046, 0.81609195, 0.70114943, 0.72413793, 0.75862069, 0.81609195, 0.8045977, 0.79310345, 0.72413793,0.57954545, 0.83908046, 0.81609195, 0.70114943, 0.72413793, 0.75862069, 0.81609195, 0.8045977,  0.79310345, 0.72413793]

dataDT = [0.69318182, 0.93103448, 0.82758621, 0.77011494, 0.75862069, 0.82758621, 0.91954023, 0.96551724, 0.75862069, 0.55172414,0.71590909, 0.91954023, 0.85057471, 0.75862069, 0.75862069, 0.8045977, 0.90804598, 0.97701149, 0.75862069, 0.51724138]

dataSvc = [0.75, 0.90804598, 0.82758621, 0.72413793, 0.81609195, 0.86206897, 0.90804598, 0.91954023, 0.81609195, 0.67816092,0.75, 0.90804598, 0.82758621, 0.72413793, 0.81609195, 0.86206897, 0.90804598, 0.91954023, 0.81609195, 0.67816092]

dataKnn = [0.65909091, 0.82758621, 0.7816092,  0.62068966, 0.66666667, 0.64367816, 0.74712644, 0.74712644, 0.66666667, 0.54022989, 0.65909091, 0.82758621, 0.7816092,  0.62068966, 0.66666667, 0.64367816, 0.74712644, 0.74712644, 0.66666667, 0.54022989]

dataAB = [0.65909091, 0.68965517, 0.85057471, 0.70114943, 0.67816092, 0.65517241, 0.91954023, 0.74712644, 0.7816092,  0.55172414, 0.65909091, 0.68965517, 0.85057471, 0.70114943, 0.67816092, 0.70114943, 0.91954023, 0.74712644, 0.71264368, 0.55172414]

dataRF = [0.75, 0.94252874, 0.87356322, 0.83908046, 0.87356322, 0.94252874, 0.96551724, 0.96551724, 0.79310345, 0.63218391, 0.73863636, 0.94252874, 0.86206897, 0.82758621, 0.87356322, 0.88505747, 0.96551724, 0.95402299, 0.79310345, 0.64367816]

print('Test between SVC and NaiveBayes')
stat, p = wilcoxon(dataSvc, dataNB)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.1
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
	
print('\nTest between SVC and Decision Tree')
stat, p = wilcoxon(dataSvc, dataDT)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.1
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
	
print('\nTest between SVC and k-NN')
stat, p = wilcoxon(dataSvc, dataKnn)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.1
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
	
print('\nTest between SVC and Adaboost')
stat, p = wilcoxon(dataSvc, dataAB)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.1
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
	
print('\nTest between SVC and Random Forest')
stat, p = wilcoxon(dataSvc, dataRF)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.1
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
	
