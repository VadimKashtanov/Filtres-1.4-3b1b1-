#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static void plume_pred(Mdl_t * mdl, uint t0, uint t1) {
	float * ancien = mdl_pred(mdl, t0, t1, 3);
	printf("PRED GENERALE = ");
	FOR(0, p, P) printf(" %f%% ", 100*ancien[p]);
	printf("\n");
	free(ancien);
};

float pourcent_masque_nulle[C] = {0};

float * pourcent_masque = de_a(0.10, 0.00, C);

//	# Un jour reflechire a f(x@p0 + b0) * f(x@p1 + b1) + f(x@p2 + b2)

float * alpha = /*de_a(1e-2, 1e-2, C);//*/de_a(1e-2, 1e-4, C);

//	## (x/3) * (x-2)**2                     ##
//	## score(x) + rnd()*abs(score(x))*0.05  ##

uint optimiser_tous_les__nulle[C] = UNIFORME_C(1);

uint optimiser_tous_les[C] = UNIFORME_C(1);/*{
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1
};*/

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	//
	//pourcent_masque[0] /= 10;
	//pourcent_masque[1] /= 10;
	alpha[0] *= 100;
	//
	MSG("S(x) Ajouter un peut d'aléatoire");
	MSG("S(x) Eventuellement faire des prediction plus lointaines");
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");   charger_tout();

	//	-- Verification --
	//titre("Verifier MDL");     verif_mdl_1e5();

	//===============
	titre("  Programme Generale  ");
	ecrire_structure_generale("structure_generale.bin");

	uint Y[C] = {
		1024,
		512,
		256,
		512,
		256,
		128,
		256,
		128,
		64,
		128,
		64,
		32,
		64,
		32,
		16,
		32,
		16,
		8,
		16,
		8,
		4,
		8,
		4,
		2,
		4,
		2,
		P
	};
	uint insts[C] = UNIFORME_C(DOT1D);//{
	insts[0] = FILTRES_PRIXS;
	//
	//	Assurances :
	ema_int_t * bloque[BLOQUES] = {
	//			    Source,      Nature,  K_ema, Intervalle, decale,     {params}
	// ----
		cree_ligne(SRC_PRIXS, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 32, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 16, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 8.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 16.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 256, 32.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 256, 256, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 32, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 16, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 8.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 16.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 256, 32.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 256, 256, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 32, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 16, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 8.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 16.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 256, 32.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 256, 256, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 32, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 16, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 8.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 16.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 256, 32.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 256, 256, 0, cree_DIRECTE()),
	// ----
		cree_ligne(SRC_PRIXS, MACD, 1, 1, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 4, 4, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 16, 1.0, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 16, 16, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 64, 4.0, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 64, 64, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 128, 8.0, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 128, 128, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 1, 1, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 4, 4, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 16, 1.0, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 16, 16, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 64, 4.0, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 64, 64, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 128, 8.0, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 128, 128, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 1, 1, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 4, 4, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 16, 1.0, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 16, 16, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 64, 4.0, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 64, 64, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 128, 8.0, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 128, 128, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 1, 1, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 4, 4, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 1.0, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 16, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 4.0, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 64, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 128, 8.0, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 128, 128, 0, cree_MACD(1)),
	// ----
		cree_ligne(SRC_HIGH, CHIFFRE, 1, 1, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 8, 8, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 32, 32, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 128, 128, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 1, 1, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 8, 8, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 32, 32, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 128, 128, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 1, 1, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 8, 8, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 32, 32, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 128, 128, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 1, 1, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 8, 8, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 32, 32, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 128, 128, 0, cree_CHIFFRE(10000))
	};
	//
	Mdl_t * mdl = cree_mdl(Y, insts, bloque);

	//Mdl_t * mdl = ouvrire_mdl("mdl.bin");

	enregistrer_les_lignes_brute(mdl, "lignes_brute.bin");

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = ROND_MODULO(FIN, (16*16));
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%(16*16)=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%(16*16));
	//
	plume_pred(mdl, t0, t1);
	//comportement(mdl, t0, t0+16*16);
	//
	srand(time(NULL));
#define PERTURBATIONS 0
	//
	uint REP = 300;
	FOR(0, rep, REP) {
		perturber(mdl, 10);
		perturber_filtres(mdl, 100);
		optimisation_mini_packet(
			mdl,
			t0, t1, 16*16*1,
			alpha, 1.0,
			RMSPROP, 40,
			pourcent_masque,
			PERTURBATIONS,
			optimiser_tous_les);
		/*optimiser(
			mdl,
			t0, t1,
			alpha, 1.0,
			RMSPROP, 150,
			//pourcent_masque_nulle);
			pourcent_masque,
			PERTURBATIONS,
			optimiser_tous_les);*/
		mdl_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, "mdl.bin");
		plume_pred(mdl, t0, t1);
		//
		printf("===================================================\n");
		printf("==================TERMINE %i/%i=======================\n", rep+1, REP);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};