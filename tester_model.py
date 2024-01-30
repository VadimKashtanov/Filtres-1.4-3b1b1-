#! /usr/bin/python3

from mdl import *

import matplotlib.pyplot as plt

signe = lambda x: (1 if x >= 0 else -1)

plusde50 = lambda x: ((x) if abs(x) >= 0.0 else 0)

prixs = I__sources[0]

if __name__ == "__main__":
	mdl = Mdl("mdl.bin")

	#	Lignes
	#print(mdl.lignes)

	print("Calcule ...")
	pred = mdl()
	print("Fin Calcule")

	_prixs = I__sources[0][DEPART:]

	print(len(pred), len(_prixs))

	plt.plot(e_norme(_prixs));
	plt.plot(pred, 'o');
	#
	plt.plot([0 for _ in pred], label='-')
	for i in range(len(pred)): plt.plot([i for _ in pred], e_norme(list(range(len(pred)))), '--')

	plt.show()

	#	Prixs && predictions
	'''plt.plot([2*x-1 for x in prixs], label='prixs')
	for i in range(I):
		s = 0
		a = [s:=(s + 0.1*signe(pred[i][j])) for j in range(len(pred[i]))]
		b = [i+j for j in range(len(pred[i]))]
		plt.plot(a, b)

	#plt.plot(pred, label='pred')

	#	Horizontale et verticales
	plt.plot([0 for _ in pred], label='-')
	for i in range(len(pred)): plt.plot([i for _ in pred], e_norme(list(range(len(pred)))), '--')'''

	#	plt
	#plt.legend()
	#plt.show()

	##	================ Gain ===============
	u = 100
	usd = []
	#
	decale = 0
	LEVIER = 1#25
	#
	I_PREDS = I_PRIXS - DEPART
	for i in range(I_PREDS-T, I_PREDS-1):
		u += u * LEVIER * 0.3 * (pred[i]) * (_prixs[i+1]/prixs[i]-1)
		if (u <= 0): u = 0
		print(f"usd = {u}")
		usd += [u]
	plt.plot(usd); plt.show()