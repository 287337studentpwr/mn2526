#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <queue>

// -------------------------------------------------- pomocnicze -------------------------------------------------

void save_csv(std::ofstream& file, int degree, double x, double exact, double approx){
    file << degree << ","
         << x << ","
         << exact << ","
         << approx << "\n";
}

std::vector<double> linspace(const std::pair<double, double> interval, const int ammount){
    double a = interval.first;
    double b = interval.second;

    if (a>b){
        std::swap(a, b);
    }
    std::vector<double> output(ammount, 0.0);
    double delta = (b - a) / (ammount-1);

    for (int i = 0 ; i<ammount; i++){
        double x = a + i*delta;
        output[i] = x;
    }
    return output;
}
// -------------------------------------------------- zad 1 -------------------------------------------------

double calculate_sum_1(int range){
    double suma = 0.0;
    for(int n = 1; n<=range; n++){
        suma += 1.0/pow(n, 2);
    }
    return suma;
}

double calculate_sum_2(int range){
    double suma = 0.0;
    for(int n = range; n>0; n--){
        suma += 1.0/pow(n,2);
    }
    return suma;
}

double calculate_sum_3(int range){
    std::priority_queue<double, std::vector<double>, std::greater<double>> pq;

    for(double n= 1.0; n<=range; n++ ){
        pq.push(1.0/pow(n, 2));
    }
    while (pq.size() >1){
        double min_1 = pq.top(); pq.pop();
        double min_2 = pq.top(); pq.pop();
        pq.push(min_2 + min_1);
    }
    return pq.top();
}

void exercise_1(){
    std::cout<< "---------------- exercise_1 ----------------" << '\n' << '\n';
    int range = 1e6;
    double sum_1 = calculate_sum_1(range);
    double sum_2 = calculate_sum_2(range);
    double sum_3 = calculate_sum_3(range);
    std::cout << "ascending order: " << sum_1 << '\n';
    std::cout << "descending order: " << sum_2 << '\n';
    std::cout << "two smallest numbers method: " << sum_3 << '\n';
}
// -------------------------------------------------- zad 2 -------------------------------------------------

/// szereg (0.4)^n do 10^6

double calculate_sum2_1(int range, double diff){
    double suma = 0.0;
    for(int n = 0; n<=range; n++){
        suma +=pow(diff,n);
    }
    return suma;
}

double calculate_sum2_2(int range, double diff){
    double suma = 0.0;
    for(int n = range; n>=0; n--){
        suma += pow(diff, n);
    }
    return suma;
}
double calculate_sum2_3(int range, double diff){
    std::priority_queue<double, std::vector<double>, std::greater<double>> pq;

    for(double n= 0.0; n<=range; n++ ){
        pq.push(pow(diff, n));
    }
    while (pq.size() >1){
        double min_1 = pq.top(); pq.pop();
        double min_2 = pq.top(); pq.pop();
        pq.push(min_2 + min_1);
    }
    return pq.top();
}


void exercise_2(){
    std::cout<< "---------------- exercise_2 ----------------" << '\n' << '\n';
    int range = 1e6;
    double diff = 0.4;
    double expected = 1.0*(1.0-pow(diff, range))/(1.0-diff);
    double sum_1 = calculate_sum2_1(range, diff);
    double sum_2 = calculate_sum2_2(range, diff);
    double sum_3 = calculate_sum2_3(range, diff);
    std::cout << "expected: " << expected << '\n';
    std::cout << "ascending order: " << sum_1 << '\n';
    std::cout << "descending order: " << sum_2 << '\n';
    std::cout << "two smallest numbers method: " << sum_3 << '\n';
}

// -------------------------------------------------- zad 3 -------------------------------------------------
double function_1(double x){
    return std::abs(x);
}


std::vector<double> get_values(std::vector<double> nodes, double (*f)(double)){
    int n = nodes.size();
    std::vector<double> values(n, 0.0);

    for (int i=0; i<=n-1; i++){
        values[i] = f(nodes[i]);
    }
    return values;
}

double lagrangeInterpolation(double x,
                             const std::vector<double>& nodes,
                             const std::vector<double>& values) {
    int n = nodes.size() - 1;
    double result = 0.0;

    for (int j = 0; j <= n; ++j) {
        double Lj = 1.0;

        for (int m = 0; m <= n; ++m) {
            if (m != j) {
                Lj *= (x - nodes[m]) / (nodes[j] - nodes[m]);
            }
        }

        result += values[j] * Lj;
    }

    return result;
}



void exercise_3(){
    std::cout<< "---------------- exercise_3 ----------------" << '\n' << '\n';

    std::ofstream file("data_lista_2_f1.csv"); // reset csv
    file << "degree,x,exact,approx\n";

    std::pair<double, double> interval = {-1,1};
    std::vector<double> test_x = linspace(interval, 1e3);
    for (int degree = 1; degree<=20; degree+=1){
        std::vector<double> nodes = linspace(interval, degree + 1);
        std::vector<double> values = get_values(nodes, function_1);

        double max_error = 0.0;

        for (double x : test_x){
            double exact = function_1(x);
            double interp = lagrangeInterpolation(x, nodes, values);
            double error = std::abs(interp - exact);

            save_csv(file, degree, x, exact, interp);

            if (error > max_error){
                max_error = error;
            }
        }
        std::cout << "degree= " << degree << '\t'
                  << "max_error= " << max_error << '\n';
    }
}

// -------------------------------------------------- zad 4 -------------------------------------------------

double function_2(const double x){
    return 1.0/(1.0 + 25.0*std::pow(x, 2));
}

std::vector<double> get_values_2(std::vector<double> nodes){
    int n = nodes.size();
    std::vector<double> values(n, 0.0);

    for (int i=0; i<=n-1; i++){
        values[i] = std::abs(nodes[i]);
    }
    return values;
}


void exercise_4(){ 
    std::cout<< "---------------- exercise_4 ----------------" << '\n' << '\n';

    std::ofstream file("data_lista_2_f2.csv"); // reset csv
    file << "degree,x,exact,approx\n";

    std::pair<double, double> interval = {-1,1};
    std::vector<double> test_x = linspace(interval, 1e3);
    for (int degree = 1; degree<=20; degree+=1){
        std::vector<double> nodes = linspace(interval, degree + 1);
        std::vector<double> values = get_values(nodes, function_2);

        double max_error = 0.0;

        for (double x : test_x){
            double exact = function_2(x);
            double interp = lagrangeInterpolation(x, nodes, values);
            double error = std::abs(interp - exact);

            save_csv(file, degree, x, exact, interp);

            if (error > max_error){
                max_error = error;
            }
        }
        std::cout << "degree= " << degree << '\t'
                  << "max_error= " << max_error << '\n';
    }
}



// -------------------------------------------------- wnioski -------------------------------------------------

void conclusions(){
    std::cout<< "---------------- wnioski ----------------" << '\n' << '\n';

    std::cout<< "1) Wyzszy stopnien nie oznacza lepszej dokladnosci." <<  '\n'
             << "2) Dla stopni nie parzystych przyblizenie jest dokladniejsze.";
}

int main(){
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();

    conclusions();

    return 0;
}