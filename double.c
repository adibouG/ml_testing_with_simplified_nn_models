#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train_data[][2] = {
    {0, 0},
    {1, 2},
    {2, 4}, 
    {3, 6}, 
    {4, 8},
};

#define train_data_length_count ((sizeof train_data)/(sizeof train_data[0]))

size_t get_train_data_length(float * td) 
{
    return ((sizeof td) / (sizeof td[0]));
}



float rand_float()
{
    return (float) rand() / (float) RAND_MAX;
}


size_t N_ITER = 500 ;


float diff_cost(float w, float b)
{    
   float dcost = 0.0f;
   for (size_t i = 0; i < train_data_length_count; ++i) {
        float in = train_data[i][0];
        float out = in * w + b;
        float d = out - train_data[i][1];
        dcost += d*d;        
    }
    dcost /= train_data_length_count;
    return dcost; 
}


int main() 
{
    srand(time(0));
    //srand(69);
    float w = rand_float()*10.0f;
    float b = 1;      
    float delta = 1e-3; //0.5f; // 1e-3; 
       
    float learn_rate = 1e-3; //0.5f; // 1e-3;
    for (size_t iter=0; iter < N_ITER; ++iter) {     
      float dcost_fd = (diff_cost(w + delta, b) - diff_cost(w, b)) / delta ;
      float dbias = (diff_cost(w, b + delta) - diff_cost(w, b)) / delta;
      w -= learn_rate * dcost_fd;
      b -= learn_rate * dbias;
      printf( "cost(w=%f, b=%f) = %f\n" , w, b, diff_cost(w, b));

    }
        
    printf( "w=%f, b=%f\n" , w, b);
 
    return 0;
}
