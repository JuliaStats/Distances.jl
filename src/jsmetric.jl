# The square root of the Jensen–Shannon divergence is a metric ,it called Jensen Shannon metric.
# Here is the Jensen-Shannon divergence of m discrete probability distributions (p^1, ...,p^m ): 
# JSmetric(p^1,...,p^m)=H(\frac{p^1+...+p^m}{m})-\frac{\sum_{j=1}^{m} H(p^j)}{m} 
# where  H() is entropy.
# We can use this function to calculate multi-distributions JS divergence square root.

type JSMetric <: Metric end

# JS metric function
 
 function evaluate(dist::JSMetric, a::AbstractArray)
     (x, y) = size(a)
     a_row = sum(a,2) / y
     entropy_average = 0
     for i = 1 : x
       @inbounds a_rowi = a_row[i]
       entropy_average += (a_rowi > 0 ? -1 * a_rowi * log(a_rowi) : 0)
     end
     average_entropy = 0
     for i = 1 : x
       for j = 1 : y
         @inbounds aij = a[i,j]
         average_entropy += (aij > 0 ? -1 * aij * log(aij) : 0)
       end
     end
     jsd = ((entropy_average -  average_entropy / y) < 1.0e-16) ? 0 : (entropy_average -  average_entropy / y)
     sqrt(jsd)
 end
 
 js_metric(a::AbstractArray) = evaluate(JSMetric(), a)
