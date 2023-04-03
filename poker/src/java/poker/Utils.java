package poker;
import java.lang.AutoCloseable;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.lang.Exception;
import clojure.lang.*;

public class Utils {
    public static void with_open(Supplier<AutoCloseable> get_resource, Consumer<AutoCloseable> body)
    throws Exception{
        try(AutoCloseable r = get_resource.get()){
            body.accept(r);
        }
    }

    public static void with_open(IFn get_resource, IFn body)
    throws Exception{
        try(AutoCloseable r = (AutoCloseable) get_resource.invoke()){
            body.invoke(r);
        }
    }

}
