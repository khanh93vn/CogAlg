#include <stdio.h>
#include <malloc.h>
#include "linked_list.h"

LinkedList test_linked_list() {
    LinkedList *ll = (LinkedList *) malloc(sizeof(LinkedList));
    ll->first = NULL;
    ll_append(ll, 12);
    ll_append(ll, 1);

    return *ll;
}

int main() {
    LinkedList a;
    ll_append(&a, 1);
    ll_append(&a, 2);
    ll_append(&a, 3);
    ll_append(&a, 4);
    printf("popped : %d\n", ll_pop(&a));
    printf("popped : %d\n", ll_pop(&a));
    printf("popped : %d\n", ll_popleft(&a));
    printf("popped : %d\n", ll_popleft(&a));
    ll_appendleft(&a, 6);
    ll_appendleft(&a, 7);
    ll_appendleft(&a, 8);
    ll_appendleft(&a, 9);
    printf("a[0] = %d\n", ll_get(a, 0));
    printf("a[3] = %d\n", ll_get(a, 3));
    printf("a[2] = %d\n", ll_get(a, 2));
    printf("a[1] = %d\n", ll_get(a, 1));

    return 0;
}
