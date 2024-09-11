
void layernorm_forward_cpu(float *out, float *mean, float *rstd,
                           const float *inp, const float *weight, const float *bias,
                           int B, int T, int C)
{
    float eps = 1e-5f;

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            const float *x = inp + b * T * C + t * C;
            float m = 0.0f;
            for (int c = 0; c < C; c++)
            {
                m += x[c];
            }
            m = m / C;
            float v = 0.0f;
            for (int c = 0; c < C; c++)
            {
                float xshift = x[c] - m;
                v += xshift * xshift;
            }
            v = v / C;
            float s = 1.0f / sqrtf(v + eps);
            float *out_pt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++)
            {
                float n = (s * (x[i] - m));
                float o = n * weight[i] + bias[i];
                out_pt[i] = o;
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

__global__ void LayerNorm(float *out, float *mean, float *rstd,
                           const float *inp, const float *weight, const float *bias,
                           int B, int T, int C)
{
    return;
}
