
template <class T>
T absDiff(T *x0, T *x1, int n)
{
    T diff= 0.0;
    for (int i= 0; i < n; i++) {
	diff+= abs(x0[i]-x1[i]);
    }
    return diff;
}
