#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

uint * cree_CHIFFRE(uint chiffre) {
	uint * ret = alloc<uint>(MAX_PARAMS);
	ret[0] = chiffre;
	return ret;
}

void nature2__chiffre(ema_int_t * ema_int) {
	//			-- Parametres --
	uint cible = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[CHIFFRE][0] <= cible && cible <= max_param[CHIFFRE][0]);
	//	-- Transformation des Parametres --
	float chiffre = (float)cible;
	//		-- Calcule de la Nature --
	_outil_chiffre(ema_int->brute, ema_int->ema, chiffre);
};