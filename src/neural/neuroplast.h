#ifndef NEUROPLAST_H
#define NEUROPLAST_H

typedef struct {
    float alpha;  // slope (pente)
    float beta;   // shift (décalage)
    float gamma;  // plateau_height (hauteur du plateau)
    float delta;  // plateau_width (largeur du plateau)
} NeuroPlastParams;

// Prototypes des fonctions
void neuroplast_init_params(NeuroPlastParams *params, float alpha, float beta, float gamma, float delta);
void neuroplast_optimize_params(NeuroPlastParams *params, float *x, float *y, int n);

// Nouvelle fonction pour obtenir la dérivée de neuroplast
float neuroplast_get_derivative(float x, NeuroPlastParams *params);

#endif // NEUROPLAST_H 