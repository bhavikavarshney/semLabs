#include <math.h>
#include <stdio.h>
// Power function to return value of a ^ b mod P
long long int power(long long int a, long long int b,
                    long long int P){
    if (b == 1)
        return a;
    else
        return (((long long int)pow(a, b)) % P);
}
// Driver program
int main(){
    long long int P, G, x, a, y, b, ka, kb;

    P = 23; // A prime number P is taken
    printf("The value of P : %lld\n", P);

    G = 9; // A primitive root for P, G is taken
    printf("The value of G : %lld\n\n", G);

    a = 4; // a is the chosen private key
    printf("The private key a for Alice : %lld\n", a);
    x = power(G, a, P); // gets the generated key
    
    b = 3; // b is the chosen private key
    printf("The private key b for Bob : %lld\n\n", b);
    y = power(G, b, P); // gets the generated key

    // Generating the secret key
    ka = power(y, a, P); // Secret key for Alice
    kb = power(x, b, P); // Secret key for Bob
    printf("Secret key for the Alice is : %lld\n", ka);
    printf("Secret Key for the Bob is : %lld\n", kb);
    return 0;
}// cc 8_diffie_hellman.c -lm
// ./a.out