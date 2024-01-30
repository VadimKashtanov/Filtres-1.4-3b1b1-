#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static float filtre(uint depart, float * x, float * f, uint intervalle, uint decale) {
	float normer_x[N];
	//
	float _min=x[depart - (decale+0)*intervalle], _max=x[depart - (decale+1)*intervalle];
	normer_x[0] = _min;
	FOR(1, i, N) {
		float a = x[depart - (decale+i)*intervalle];
		normer_x[i] = a;
		if (a > _max) _max = a;
		if (a < _min) _min = a;
	}
	FOR(0, i, N) normer_x[i] = (normer_x[i]-_min)/(_max-_min);
	//
	float s = 0, d = 0;
	float f_nouveau = f[0];
	float x_nouveau = normer_x[0];
	s += sqrtf(1 + fabs(x_nouveau - f_nouveau));
	float f_avant = f_nouveau;
	float x_avant = x_nouveau;
	FOR(1, i, N) {
		f_nouveau = f[i];
		x_nouveau = normer_x[i];
		s += sqrtf(1 + fabs(  x_nouveau   -   f_nouveau  ));
		d += powf((1 + fabs((x_nouveau-x_avant) - (f_nouveau-f_avant))), 2);
		f_avant   = f_nouveau;
		x_avant   = x_nouveau;
	};

	s = s/8-1;
	d = d/7-1;

	return 2*expf(-s*s -d*d)-1;
};


int main(int argc, char ** argv) {
	srand(0);
	cudaSetDevice(0);
	//
	FILE * fp = fopen(argv[1], "rb");
	//
	uint Y[C];
	FREAD(Y, sizeof(uint), C, fp);
	//
	uint PRIXS_bitget;
	FREAD(&PRIXS_bitget, sizeof(uint), 1, fp);
	uint intervalles[BLOQUES], decales[BLOQUES];
	FREAD(intervalles, sizeof(uint), BLOQUES, fp);
	FREAD(decales,     sizeof(uint), BLOQUES, fp);
	//
	float * lignes = alloc<float>(PRIXS_bitget*BLOQUES);
	FREAD(lignes, sizeof(float), PRIXS_bitget*BLOQUES, fp);
	//
	float * poids[C];
	FOR(0, c, C) {
		uint POIDS;
		FREAD(&POIDS, sizeof(uint), 1, fp);
		poids[c] = alloc<float>(POIDS);
		FREAD(poids[c], sizeof(float), POIDS, fp);
	}
	//
	fclose(fp);

	//	------------- Calcule ----------------
	float * y_avant   = alloc<float>( PRIXS_bitget*MAX_Y );
	float * y_nouveau = alloc<float>( PRIXS_bitget*MAX_Y );
	//
	/*#pragma omp parallel
	#pragma omp for*/
	FOR(0, f, BLOQUES*F_PAR_BLOQUES) {
		uint b = (f - (f % F_PAR_BLOQUES)) / F_PAR_BLOQUES;
		FOR(DEPART, t, PRIXS_bitget) {
			y_nouveau[t*MAX_Y + f] = filtre(
				b*PRIXS_bitget + t,
				lignes,
				poids[0] + f*N,
				intervalles[b], decales[b]
			);
		}
	};
	/*#pragma omp parallel
	#pragma omp for*/
	FOR(0, i, PRIXS_bitget*MAX_Y) y_avant[i] = y_nouveau[i];
	//
	FOR(1, c, C) {
		uint X = Y[c-1];
		/*#pragma omp parallel
		#pragma omp for*/
		FOR(0, i, Y[c]) {
			FOR(DEPART, t, PRIXS_bitget) {
				float s = poids[c][(X+1)*i + X-1+1];
				FOR(0, j, X) s += poids[c][(X+1)*i + j] * y_avant[t*MAX_Y + j];
				y_nouveau[t*MAX_Y + i] = tanh(s);
			};
		};

		/*#pragma omp parallel
		#pragma omp for*/
		FOR(0, i, PRIXS_bitget*MAX_Y) y_avant[i] = y_nouveau[i];
	};

	//	---------- Ecrire Resultat ----------
	fp = fopen(argv[1], "wb");
	float res[PRIXS_bitget];
	FOR(DEPART, t, PRIXS_bitget) res[t] = y_nouveau[t*MAX_Y + 0];
	FWRITE(res+DEPART, sizeof(float), (PRIXS_bitget-DEPART), fp);
	fclose(fp);
}