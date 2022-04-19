#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void get_parameters_from_csv_line(char* line, double* parameter_set){
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


void get_settings_from_csv_line(char* line, double* setting_vals){
    int i = 0;
    char* tok;
    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")){
        double val;
        val = strtod(tok,NULL);
        setting_vals[i] = val;
        i++;
        if(i>=3)
            return;
            
    }
    return;
}

void simulate_WC(int ts_size, double* initial_conditions, double* params, FILE* output_tsE, FILE* output_tsI, double dt,int circ_array_size){
    //clock_t begin=clock();

    static double time_series_E[10000000];
    static double time_series_I[10000000];
    
    double time_series_E_temp;
    double time_series_I_temp;

    double time_series_E_corr_pt;
    double time_series_I_corr_pt;
    int i;
    int next_i;
    int dumped_up_to = 0;
    circ_array_size = 10000000;
    int remaining = (ts_size%circ_array_size);
    if (remaining == 0) remaining = circ_array_size;
    //printf("%d\n",remaining);
    time_series_E[0] = initial_conditions[0];
    time_series_I[0] = initial_conditions[1];
    
    
    
    for(int ii = 0; ii<ts_size-1; ii++){
        i = ii%circ_array_size;
        next_i = (ii+1)%circ_array_size;

        if (next_i == 0 && ii > 0){
            for(int j = 0; j<circ_array_size;j++){
                // printf("%d E: %lf, I: %lf\n",i,time_series_E[i],time_series_I[i]);
                fprintf(output_tsE,"%lf%s",time_series_E[j], (j<circ_array_size?",":""));
                fprintf(output_tsI,"%lf%s",time_series_I[j], (j<circ_array_size?",":""));
            }
            dumped_up_to+=circ_array_size;
        }
        
        time_series_E[next_i] = params[0] * time_series_E[i] - params[2] * time_series_I[i] + params[18] - params[9];
        time_series_I[next_i] = params[1] * time_series_E[i] - params[3] * time_series_I[i] + params[19] - params[13];
        time_series_E[next_i] = params[8] / (1.0 + exp(-params[6]* (params[20] * time_series_E[next_i] - params[7])));
        time_series_I[next_i] = params[12] / (1.0 + exp(-params[10]* (params[21] * time_series_I[next_i] - params[11])));
        time_series_E[next_i] = dt*(((params[16] - params[14] * time_series_E[i]) * time_series_E[next_i]) - time_series_E[i]) / params[4];
        time_series_I[next_i] = dt*(((params[17] - params[15] * time_series_I[i]) * time_series_I[next_i]) - time_series_I[i]) / params[5];
        time_series_E_temp = time_series_E[i] + time_series_E[next_i]; 
        time_series_I_temp = time_series_I[i] + time_series_I[next_i];
        // Corrector point
        time_series_E_corr_pt = params[0] * time_series_E_temp - params[2] * time_series_I_temp + params[18] - params[9];
        time_series_I_corr_pt = params[1] * time_series_E_temp - params[3] * time_series_I_temp + params[19] - params[13];
        time_series_E_corr_pt = params[8] / (1.0 + exp(-params[6] * (params[20] * time_series_E_corr_pt - params[7])));
        time_series_I_corr_pt = params[12] / (1.0 + exp(-params[10] * (params[21] * time_series_I_corr_pt - params[11])));
        time_series_E_corr_pt = dt*(((params[16] - params[14] * time_series_E_temp) * time_series_E_corr_pt) - time_series_E_temp) / params[4]; 
        time_series_I_corr_pt = dt*(((params[17] - params[15] * time_series_I_temp) * time_series_I_corr_pt) - time_series_I_temp) / params[5];
        //Heun point
        time_series_E[next_i] = time_series_E[i] + (time_series_E[next_i]+time_series_E_corr_pt)/2; 
        time_series_I[next_i] = time_series_I[i] + (time_series_I[next_i]+time_series_I_corr_pt)/2;
        //printf("%d E: %lf, I: %lf\n",i,time_series_E[next_i],time_series_I[next_i]);
        // printf("\n\n");
        
         
    }
    
    for(int j = 0; j<remaining;j++){
        // printf("%d E: %lf, I: %lf\n",i,time_series_E[i],time_series_I[i]);
        fprintf(output_tsE,"%lf%s",time_series_E[j], (j<circ_array_size?",":""));
        fprintf(output_tsI,"%lf%s",time_series_I[j], (j<circ_array_size?",":""));
    }

    fprintf(output_tsE,"\n");
    fprintf(output_tsI,"\n");
    //clock_t end=clock();
    //printf("Time taken:%lf",(double)(end-begin)/CLOCKS_PER_SEC);
    //printf("Done");
    return;
}

void run_simulation_set(int num_sim,int ts_size, double* initial_conditions, FILE* stream,FILE* output_tsE, FILE* output_tsI, double dt, double length){
    double params[num_sim][22];
    char line[1024];
    int sim_count = 0;
    int circ_array_size = 120000;
    if (ts_size<circ_array_size) circ_array_size = ts_size;
    stream = fopen("./Simulations/Input_parameters/parameters.csv", "r");
    while (fgets(line, 1024, stream)){
        char* tmp = strdup(line);
        get_parameters_from_csv_line(tmp, params[sim_count]);
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
    printf("Number of simulations: %d, length: %lf\n",num_sim,length);
    
    for(int i = 0; i<num_sim;i++){
        simulate_WC(ts_size, initial_conditions, params[i], output_tsE, output_tsI,dt,circ_array_size);
    }
    return;
}


int main(int argc, char *argv[]){
    double length;
    double dt;
    int ts_size;
    double initial_conditions[2];
    double setting_vals[3];
    int num_sim;
    FILE* stream; //= fopen(argv[1], "r");
    FILE* output_tsE;
    FILE* output_tsI;
    char line[1024];
    FILE* settings = fopen("./Simulations/Input_parameters/settings.csv", "r");
    if(fgets(line, 1024, settings)) {
        get_settings_from_csv_line(line, setting_vals);
   }
   
    initial_conditions[0] = 0.25;
    initial_conditions[1] = 0.25;

    length = setting_vals[0];
    dt = setting_vals[1];
    ts_size = (int)(1000*length/dt);
    num_sim = (int) setting_vals[2];
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
    clock_t begin=clock();
    run_simulation_set(num_sim,ts_size, initial_conditions, stream, output_tsE, output_tsI, dt, length);
    clock_t end=clock();
    printf("Time taken:%lf",(double)(end-begin)/CLOCKS_PER_SEC);
    fclose(output_tsE);
    fclose(output_tsI);
    
}