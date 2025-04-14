
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
// AND
float train_data[][3] = {
  {0, 0, 0},
  {1, 0, 0},
  {0, 1, 0}, 
  {1,1,1 }, 
  
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


size_t N_ITER = 5000 ;

//activation fn: Relu , sigmoid, ...

float sigm(float c)
{
  return 1.f / (1.f + expf(-c));
}


float diff_cost(float w1, float w2, float b)
{    
   float dcost = 0.0f;
   for (size_t i = 0; i < train_data_length_count; ++i) {
        float in1 = train_data[i][0];
	float in2 = train_data[i][1];
        float out = sigm(in1 * w1 +  in2 * w2 + b);
        float d = out - train_data[i][2];
        dcost += d*d;        
    }
    dcost /= train_data_length_count;
    return dcost; 
}


int main() 
{
    srand(time(0));
    //srand(69);
    float w1 = rand_float()*10.0f  ;  
    float w2 = rand_float()*10.0f;
    float b = rand_float()*0.0f;      
    float delta = 1e-1; //0.5f; // 1e-3; 
       
    float learn_rate = 1e-1; //0.5f; // 1e-3;
    for (size_t iter=0; iter < N_ITER; ++iter) {     
      float dw1 = (diff_cost(w1 + delta, w2,  b) - diff_cost(w1,w2, b)) / delta;
      float dw2 = (diff_cost(w1, w2 + delta, b) - diff_cost(w1,w2, b)) / delta;
      float dbias = (diff_cost(w1, w2, b + delta) - diff_cost(w1,w2, b)) / delta;
      w1 -= learn_rate * dw1;
      w2 -= learn_rate * dw2;
      b -= learn_rate * dbias;
      printf( "cost(w1=%f, w2=%f,  b=%f) = %f\n" , w1, w2,  b, diff_cost(w1, w2, b));

    }
        
    printf( "w1=%f,w2=%f, b=%f\n" , w1, w2, b);

    // test the model after train
    for(size_t i=0; i<2; ++i){
      for (size_t j=0; j<2; ++j){
	printf("%zu|%zu = %f\n", i, j,  sigm(i*w1 + j *w2 + b));
      }
    }
 
    return 0;
}
