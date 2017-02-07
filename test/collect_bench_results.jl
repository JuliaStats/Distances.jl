# A simple script to collect benchmark results to a CSV file

#   colwise_bench_results.txt   => colwise_bench_results.csv
#   pairwise_bench_results.txt  => pairwise_bench_results.csv
#

function parse_results(lines, title1, title2)
    # just a handy function, without a lot of error handling stuff
    # assuming everything is correct

    records = Vector{Tuple{String, Float64, Float64, Float64}}(0)

    state = 0

    for raw_line in lines
        line = strip(raw_line)
        if isempty(line) || beginswith(line, "#")
            continue
        end

        if beginswith(line, "bench")
            @assert state == 0
            m = match(r"^bench\s+(\w+)", line)
            name = m.captures[1]
            state = 1

        elseif beginswith(line, title1)
            @assert state == 1
            m = match(r"^(\w+):\s+t\s*=\s*([.\d]+)", line)
            t = m.captures[1]
            @assert t == title1
            v1 = float64(m.captures[2])
            state = 2

        elseif beginswith(line, title2)
            @assert state == 2
            m = match(r"^(\w+):\s+t\s*=\s*([.\d]+)s\s+\|\s*gain\s*=\s*([.\d]+)", line)
            t = m.captures[1]
            @assert t == title2
            v2 = float64(m.captures[2])
            gain = float64(m.captures[3])

            # store the current record
            push!(records, (name, v1, v2, gain))
            state = 0
        end
    end

    return records
end


function collect(title::String, title1, title2)
    println("Processing $title ...")
    infile = string(title, ".txt")
    outfile = string(title, ".csv")

    if !isfile(infile)
        println("File $infile not found -- ignore.")
        return
    end

    fin = open(infile, "r")
    lines = try
        readlines(fin)
    catch err
        close(fin)
        throw(err)
    end

    R = parse_results(lines, title1, title2)

    fout = open(outfile, "w")
    try
        write(fout, "| name | $title1 | $title2 | gain |\n")
        for r in R
            write(fout, "| $(r[1]) | $(r[2]) | $(r[3]) | $(r[4]) |\n")
        end
        flush(fout)
    catch err
        close(fout)
        throw(err)
    end
end


# main

collect("colwise_bench_results", "loop", "colwise")
collect("pairwise_bench_results", "loop", "pairwise")


