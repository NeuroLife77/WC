#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void getfield(char* line, double* parameter_set){
    int i = 0;
    char* tok;
    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")){
        double val;
        val = strtod(tok,NULL);
        parameter_set[i] = val;
        i++;
        if(i>=22)
            return;
            
    }
    return;
}

// This one is not faster than the other one
int main(int argc, char *argv[]){
    double length = 302.0;
    
    double dt = 1.0;
    int num_sim = 20;
    int ts_size = (int)(1000*length/dt);
    printf("%d",ts_size);
    double initial_conditions[2];
    printf("Initialized settings\n");
    
    static double time_series_E[20][302000];
    static double time_series_I[20][302000];
    
    double time_series_E_temp[num_sim];
    double time_series_I_temp[num_sim];

    double time_series_E_corr_pt[num_sim];
    double time_series_I_corr_pt[num_sim];

    printf("Initialized timeseries\n");
    double params[num_sim][22];
    int sim_count = 0;
    
    // Load the parameter set
    FILE* stream = fopen("./Simulations/Input_parameters/parameters.csv", "r");
    //FILE* stream = fopen(argv[1], "r");
    FILE* output_tsE;
    FILE* output_tsI;
    char line[1024];
    clock_t begin=clock();
    
    initial_conditions[0] = 0.25;
    initial_conditions[1] = 0.25;
    while (fgets(line, 1024, stream)){
        char* tmp = strdup(line);
        getfield(tmp, params[sim_count]);
        /* print statement for debugging to make sure the right parameters are loaded
        printf("Source Added parameter ");
        for(int i = 0; i<22; i++){
            printf("%1f, ", parameters[sim_count][i]);
        }
        printf("\n");
        */
        free(tmp);
        sim_count++;
        if (sim_count>=num_sim) break;
    }
    if (sim_count<num_sim) num_sim = sim_count;
    printf("Number of simulations: %d, length: %lf",num_sim,length);
    for(int i = 0; i<num_sim; i++){
        time_series_E[num_sim][0] = initial_conditions[0];
        time_series_I[num_sim][0] = initial_conditions[1];
    }
    
    for(int i = 0; i<ts_size-1; i++){
        for(int j = 0; j<num_sim; j++){

            time_series_E[j][i+1] = params[j][0] * time_series_E[j][i] - params[j][2] * time_series_I[j][i] + params[j][18] - params[j][9];
            time_series_I[j][i+1] = params[j][1] * time_series_E[j][i] - params[j][3] * time_series_I[j][i] + params[j][19] - params[j][13];
            time_series_E[j][i+1] = params[j][8] / (1.0 + exp(-params[j][6]* (params[j][20] * time_series_E[j][i+1] - params[j][7])));
            time_series_I[j][i+1] = params[j][12] / (1.0 + exp(-params[j][10]* (params[j][21] * time_series_I[j][i+1] - params[j][11])));
            time_series_E[j][i+1] = dt*(((params[j][16] - params[j][14] * time_series_E[j][i]) * time_series_E[j][i+1]) - time_series_E[j][i]) / params[j][4];
            time_series_I[j][i+1] = dt*(((params[j][17] - params[j][15] * time_series_I[j][i]) * time_series_I[j][i+1]) - time_series_I[j][i]) / params[j][5];
            time_series_E_temp[j] = time_series_E[j][i] + time_series_E[j][i+1]; 
            time_series_I_temp[j] = time_series_I[j][i] + time_series_I[j][i+1];
            // Corrector point
            time_series_E_corr_pt[j] = params[j][0] * time_series_E_temp[j] - params[j][2] * time_series_I_temp[j] + params[j][18] - params[j][9];
            time_series_I_corr_pt[j] = params[j][1] * time_series_E_temp[j] - params[j][3] * time_series_I_temp[j] + params[j][19] - params[j][13];
            time_series_E_corr_pt[j] = params[j][8] / (1.0 + exp(-params[j][6] * (params[j][20] * time_series_E_corr_pt[j] - params[j][7])));
            time_series_I_corr_pt[j] = params[j][12] / (1.0 + exp(-params[j][10] * (params[j][21] * time_series_I_corr_pt[j] - params[j][11])));
            time_series_E_corr_pt[j] = dt*(((params[j][16] - params[j][14] * time_series_E_temp[j]) * time_series_E_corr_pt[j]) - time_series_E_temp[j]) / params[j][4]; 
            time_series_I_corr_pt[j] = dt*(((params[j][17] - params[j][15] * time_series_I_temp[j]) * time_series_I_corr_pt[j]) - time_series_I_temp[j]) / params[j][5];
            //Heun point
            time_series_E[j][i+1] = time_series_E[j][i] + (time_series_E[j][i+1]+time_series_E_corr_pt[j])/2; 
            time_series_I[j][i+1] = time_series_I[j][i] + (time_series_I[j][i+1]+time_series_I_corr_pt[j])/2;
            //printf("%d E: %lf, I: %lf\n",i,time_series_E[j][i+1],time_series_I[j][i+1]);
         }
    }
    /* Debugging purposes
    for(int i = 0; i<ts_size-1; i++){
        for(int j = 0; j<num_sim; j++){
            printf("%d E: %lf, I: %lf\n",i,time_series_E[j][i],time_series_I[j][i]);
         }
    }
    */
    output_tsE = fopen("./Simulations/Output_timeseries/simulation_testE.csv","wb");
    if (output_tsE == NULL){
        printf("Error while opening the file E.\n");
        return 0;
    }
    output_tsI = fopen("./Simulations/Output_timeseries/simulation_testI.csv","wb");
    if (output_tsI == NULL){
        printf("Error while opening the file I.\n");
        return 0;
    }
    // printf("\n\n");
    for(int i = 0; i<num_sim; i++){
        for(int j = 0; j<ts_size;j++){
            // printf("%d E: %lf, I: %lf\n",i,time_series_E[i][j],time_series_I[i][j]);
            fprintf(output_tsE,"%lf%s",time_series_E[i][j], (j<ts_size?",":""));
            fprintf(output_tsI,"%lf%s",time_series_I[i][j], (j<ts_size?",":""));
        }
        fprintf(output_tsE,"\n");
        fprintf(output_tsI,"\n");
    }
    fclose(output_tsE);
    fclose(output_tsI);
    clock_t end=clock();
    printf("Time taken:%lf",(double)(end-begin)/CLOCKS_PER_SEC);
    return 0;
}