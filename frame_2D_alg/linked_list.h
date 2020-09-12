#include <malloc.h>

#define ESIZE       12  /* size of LinkedListElement */
#define INVALID     0xFFFFFFFF

struct LinkedListElement {
    long long val;
    struct LinkedListElement *next;
};

typedef struct LinkedListElement LinkedListElement;

typedef struct {
    LinkedListElement *first;
} LinkedList;


void ll_init(LinkedList *ll, long long val)
/**
 * Insert first element at the end of a linked list.
 * Just make sure the list is empty.
 * @param ll : pointer of the linked list.
 * @param val : value to insert.
 */
{
    ll->first = (LinkedListElement *) malloc(ESIZE);
    ll->first->val = val;
    ll->first->next = NULL;
}

void ll_append(LinkedList *ll, long long val)
/**
 * Insert an element at the end of a linked list.
 * @param ll : pointer of the linked list.
 * @param val : value to insert.
 */
{
    if(ll->first == NULL) {
        ll_init(ll, val);
        return;
    }

    LinkedListElement *current = ll->first;
    while(current->next != NULL) current = current->next;
    current->next = (LinkedListElement *) malloc(ESIZE);
    current->next->val = val;
    current->next->next = NULL;
}

long long ll_pop(LinkedList *ll)
/**
 * Remove and return last element in a linked list.
 * @param ll : pointer of the linked list.
 * @return last element of the linked list.
 */
{
    if(ll->first == NULL) {
        printf("\nError: linked list is empty\n");
        return INVALID;
    }

    LinkedListElement *prev = ll->first,
                      *current = ll->first->next;

    if(current == NULL) {
        ll->first = NULL;
        current = prev;
    }
    else {
        while(current->next != NULL) {
            prev = current;
            current = current->next;
        }
        prev->next = NULL;
    }
    long long val = current->val;
    free(current);
    return val;
}

void ll_appendleft(LinkedList *ll, long long val)
/**
 * Insert an element at the start of a linked list.
 * @param ll : pointer of the linked list.
 * @param val : value to insert.
 */
{
    if(ll->first == NULL) {
        ll_init(ll, val);
        return;
    }

    LinkedListElement *new_ll = (LinkedListElement *) malloc(ESIZE);
    new_ll->val = val;
    new_ll->next = ll->first;
    ll->first = new_ll;
}


long long ll_popleft(LinkedList *ll)
/**
 * Remove and return first element in a linked list.
 * @param ll : pointer of the linked list.
 * @return last element of the linked list.
 */
{
    if(ll->first == NULL) {
        printf("\nError: linked list is empty\n");
        return INVALID;
    }

    LinkedListElement *popped = ll->first;

    ll->first = popped->next;
    long long val = popped->val;
    free(popped);
    return val;
}

long long ll_get(LinkedList ll, long i) {
    LinkedListElement *current = ll.first;
    while(i-- > 0) current = current->next;
    return current->val;
}
