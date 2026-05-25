/*
#include "matplotlibcpp.h"
#define WITHOUT_NUMPY 
namespace plt = matplotlibcpp; */

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

// -------------------------------------------------- utility -------------------------------------------------

void print_vector(std::vector<double>& vector){
    for(int i = 0; i< vector.size(); i++){
        std::cout << vector[i] << " ";
    }
    std::cout << '\n';
}

void print_pair(std::pair<double, double> pair){
    std::cout << pair.first << " " << pair.second << '\n';
}

// -------------------------------------------------- zad 1 -------------------------------------------------
std::vector<double> find_epsilons(){
    std::vector<double> output;
    double a = 1.0L;
    while(a <= 1e4){
        double epsilon = 0.5L;
        int i = 1;
        while(a + epsilon > a){
            epsilon*=0.5L;
            i++;
        }
        output.push_back(std::pow(0.5L, i));
        a*=10.0L;
    }
    return output;
}

// -------------------------------------------------- zad 2 -------------------------------------------------

 std::pair<double, double> recurrance_calc(int N, std::pair<double, double> series, std::vector<double> reccurance ){
    double a = reccurance[0];
    double b = reccurance[1];
    double c = reccurance[2];
    
    for(int i = 2; i<N; i++){
        double x  = ((-b)*series.second + (-a)*series.first)/c;
        std::swap(series.first, series.second);
        series.second = x;
    }
    for(int i = 2; i<N; i++){
        double x = ((-b)*series.first + (-c)*series.second)/a;
        std::swap(series.second, series.first);
        series.first = x;
    }
    return series;
}

std::pair<double, double> compare_diff(std::pair<double, double>& pair1, std::pair<double, double> pair2){
    std::pair<double, double> output = {0.0L, 0.0L};
    output.first = std::abs(pair1.first - pair2.first);
    output.second = std::abs(pair1.second - pair2.second);
    return output;
}

// -------------------------------------------------- zad 3 -------------------------------------------------

std::vector<std::pair<double, double>> calculate_func(){
    std::vector<std::pair<double, double>> output(6, {0.0L, 0.0L});
    for(int i = 4; i<10; i++){
        double x = std::pow(10.0L, i);
        output[i-4].first = x - sqrt(1.0L + std::pow(x, 2));
        output[i-4].second  = -1/(1+sqrt(1.0L + std::pow(x, 2)));
    }
    return output;
}

void compare_diff_func(std::vector<std::pair<double, double>>& V){
    for(int i = 0; i<V.size(); i++){
        double diff = abs(V[i].first - V[i].second);
        double f_1 = V[i].first;
        double f_2 = V[i].second;
        std::cout << f_1 << " " << f_2 << " difference: " << diff << '\n';
    }
}

// -------------------------------------------------- zad 4 -------------------------------------------------

std::vector<std::pair<double, std::pair<double, double>>> calculate_function_ex4(double x_1, double x_2 , int ammount){
    std::vector<std::pair<double, std::pair<double, double>>> output(ammount,{0.0L, {0.0L, 0.0L}});
    double  delta = (x_1 - x_2) / (ammount - 1);
    for(int i = 0; i<ammount; i++){
        double x = x_1 + i * delta;
        double g_1 = std::pow(x - 1.0L, 4);
        double g_2 = std::pow(x, 4) - 4.0L*std::pow(x, 3) + 6.0L*std::pow(x, 2) - 4.0L*x + 1.0L;
        std::pair<double, double> pair = {g_1, g_2};
        output[i] = {x, pair};
    }

    return output;
}

void save_csv(
    const std::vector<std::pair<double, std::pair<double,double>>>& data
){
    std::ofstream file("data_lista_1.csv");

    file << "x,g1,g2\n";

    for(const auto& p : data){
        file << p.first << ","
             << p.second.first << ","
             << p.second.second << "\n";
    }
}

int main(){
    // zad 1
    std::cout << "exercise 1:\n";
    std::vector<double> epsilons = find_epsilons();
    print_vector(epsilons);
    // zad 2
    std::cout << "exercise 2:\n";
    std::pair<double, double> values_before_x = {1.0L, 0.2L};
    std::vector<double> recurrance_x = {5.0L, -26.0L, 5.0L};
    std::pair<double, double> values_after_x = recurrance_calc(30, values_before_x, recurrance_x);
    std::pair<double, double> values_diff_x = compare_diff(values_before_x, values_after_x);
    std::cout << "reccurance for x:\n";
    print_pair(values_before_x);
    print_pair(values_after_x);
    print_pair(values_diff_x);
    std::vector<double> recurrance_y = {2.0L, -5.0L, 2.0L};
    std::pair<double, double> values_before_y = {1.0L, 0.5L};
    std::pair<double, double> values_after_y = recurrance_calc(30, values_before_y, recurrance_y);
    std::pair<double, double> values_diff_y = compare_diff(values_before_y, values_after_y);
    std::cout << "reccurance for y:\n";
    print_pair(values_before_y);
    print_pair(values_after_y);
    print_pair(values_diff_y);
    // zad 3
    std::cout << "exercise 3:\n";
    std::vector<std::pair<double, double>> V = calculate_func();
    compare_diff_func(V);
    // zad 4
    double x_1 = 1 - std::pow(10, -3);
    double x_2 = 1 + std::pow(10, -3);
    int data_amm = 100;
    std::vector<std::pair<double, std::pair<double, double>>> data = calculate_function_ex4(x_1, x_2, data_amm);
    save_csv(data);
    // aby splotowac wystarczy wygenerowac dane a potem odpalic plot_lista_1.py od wygenerowanych danych, wykresy nie roznia sie praktycznie wcale.

    return 0;
}
